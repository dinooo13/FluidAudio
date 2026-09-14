import Foundation

/// The audio received so far, behind the protocol `ChunkProcessor` reads
/// through, so a `ChunkProcessor` built on it sees the prefix that exists now.
final class GrowingAudioSampleSource: AudioSampleSource, @unchecked Sendable {
    private let lock = NSLock()
    private var samples: [Float] = []

    var sampleCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return samples.count
    }

    func append(_ newSamples: [Float]) {
        guard !newSamples.isEmpty else { return }
        lock.lock()
        defer { lock.unlock() }
        samples.append(contentsOf: newSamples)
    }

    func removeAll() {
        lock.lock()
        defer { lock.unlock() }
        samples.removeAll(keepingCapacity: false)
    }

    func copySamples(
        into destination: UnsafeMutablePointer<Float>,
        offset: Int,
        count: Int
    ) throws {
        lock.lock()
        defer { lock.unlock() }
        try ArrayAudioSampleSource(samples: samples)
            .copySamples(into: destination, offset: offset, count: count)
    }
}

/// `ChunkProcessor` for audio that arrives over time. Windows run as soon as
/// the audio they need exists; `finish()` runs the last window and the same
/// merge `ChunkProcessor.process` runs, so the text is identical to a batch
/// transcription of the same samples.
///
/// That identity holds because `ChunkProcessor` starts every window from a
/// fresh decoder state, so a window depends only on its own samples, and its
/// start rule for window *n+1* reads no audio past the end of window *n* (see
/// `nextSilenceAlignedStart`). Every window but the last can therefore run
/// while the recording is still going, and the merge sees the same windows in
/// the same order a batch run would have produced. One window runs at a time;
/// there is no worker pool.
public actor IncrementalChunkProcessor {

    private let manager: AsrManager
    private let language: Language?
    private let source = GrowingAudioSampleSource()

    private let decoderLayers: Int
    private let endAligned: Bool
    private let preferSilenceAlignment: Bool
    private let layout:
        (
            chunkSamples: Int, strideSamples: Int, melContextSamples: Int,
            warmupPrefixSamples: Int
        )

    /// Finished windows in order, ready for the shared merge.
    private var windows: [[ChunkProcessor.TokenWindow]] = []
    private var chunkIndex = 0
    private var previousStart = 0
    /// Start of window `chunkIndex`, once enough audio exists to decide it.
    private var pendingDecision: ChunkProcessor.ChunkStartDecision?

    /// Encoder and decoder time across every window, not wall clock.
    private var accumulatedProcessingTime: TimeInterval = 0

    private var isRunningWindow = false
    private var isCancelled = false
    private var isFinished = false

    public init(manager: AsrManager, language: Language? = nil) async throws {
        let melChunkContext = await manager.melChunkContext
        let modelVersion = await manager.modelVersion
        let dualDecodeArbitration = await manager.dualDecodeArbitration

        // `process` routes v3/no-mel to a two-decode path with its own
        // window bookkeeping.
        if dualDecodeArbitration, !melChunkContext, modelVersion == .v3 {
            throw ASRError.processingFailed(
                "IncrementalChunkProcessor does not support dualDecodeArbitration"
            )
        }

        self.manager = manager
        self.language = language
        self.decoderLayers = await manager.decoderLayerCount
        self.endAligned = ChunkProcessor.supportsSuppressedPrefix(modelVersion)
        self.preferSilenceAlignment = !melChunkContext && modelVersion == .v3

        let chunker = ChunkProcessor(sampleSource: self.source)
        self.layout = chunker.chunkLayout(
            melChunkContext: melChunkContext,
            modelVersion: modelVersion
        )
    }

    /// Appends samples and runs every window whose end is now covered.
    public func append(_ samples: [Float]) async throws {
        guard !isCancelled, !isFinished else { return }
        source.append(samples)
        try await runReadyWindows()
    }

    /// Runs the final window, merges, and returns the result.
    public func finish() async throws -> ASRResult {
        guard !isCancelled else { throw CancellationError() }
        isFinished = true
        // A caller that did not await `append` may still have a window in
        // flight; let it drain so windows stay ordered and one at a time.
        while isRunningWindow {
            await Task.yield()
        }

        let finishStarted = Date()
        let chunker = ChunkProcessor(sampleSource: source)
        let totalSamples = chunker.totalSamples
        // Audio this short never reaches `ChunkProcessor` in a batch run:
        // `transcribeWithState` frame-aligns the whole buffer and makes one
        // padded pass, where a window over the same audio declares a length
        // trimmed to the speech end and decodes differently — on a ten-second
        // fixture that pass writes "seven," and the window writes "7,". So take
        // the same short path; `runReadyWindows` holds off below this length,
        // so there is never a window to discard.
        if totalSamples <= ASRConstants.maxModelSamples {
            return try await transcribeWholeBuffer(in: chunker, totalSamples: totalSamples)
        }

        let speechEnd = endAligned ? try chunker.speechEndSamples() : totalSamples

        // `process` keeps looping until a window reports itself last. Normally
        // exactly one window is left here, but a start that could not be
        // decided while the audio was still arriving can leave two.
        while !isCancelled {
            guard
                let decision = try nextStart(in: chunker, isFinal: true),
                decision.start < totalSamples,
                let plan = planWindow(for: decision, in: chunker, speechEnd: speechEnd)
            else { break }

            try await run(plan, in: chunker)
            if plan.isLastChunk { break }
        }

        guard !isCancelled else { throw CancellationError() }

        let elapsed = Date().timeIntervalSince(finishStarted)
        return try await chunker.finalize(
            chunkOutputs: windows,
            using: manager,
            decoderLayers: decoderLayers,
            maxModelSamples: ASRConstants.maxModelSamples,
            language: language,
            // `finalize` reports `Date().timeIntervalSince(startTime)`; back-date
            // it so the figure is compute, not the seconds the speaker took.
            startTime: Date().addingTimeInterval(-(accumulatedProcessingTime + elapsed))
        )
    }

    public func cancel() {
        isCancelled = true
        windows.removeAll()
        pendingDecision = nil
        source.removeAll()
    }

    private func transcribeWholeBuffer(
        in chunker: ChunkProcessor,
        totalSamples: Int
    ) async throws -> ASRResult {
        let samples = try chunker.readSamples(offset: 0, count: totalSamples)
        // `transcribe` rejects audio below 0.3 s outright; pad rather than fail.
        let padded = manager.padAudioIfNeeded(
            samples,
            targetLength: ASRConstants.minimumRequiredSamples(forSampleRate: ASRConstants.sampleRate)
        )
        var state = TdtDecoderState.make(decoderLayers: decoderLayers)
        state.reset()
        return try await manager.transcribe(padded, decoderState: &state, language: language)
    }

    // MARK: - Window scheduling

    private func runReadyWindows() async throws {
        guard !isRunningWindow else { return }
        isRunningWindow = true
        defer { isRunningWindow = false }

        while !isCancelled, !isFinished {
            let chunker = ChunkProcessor(sampleSource: source)
            let available = chunker.totalSamples
            // Below this length `finish()` transcribes the whole buffer instead.
            guard available > ASRConstants.maxModelSamples else { return }
            guard let decision = try nextStart(in: chunker, isFinal: false) else { return }

            // Batch marks a window last when `chunkStart + visibleChunkSamples
            // >= totalSamples`, and the final total is at least `available`.
            // Strictly greater is therefore the test for "this window is
            // certainly not the last one", which is the only kind that can run
            // before the recording ends.
            let defaultWarmupSamples =
                chunkIndex > 0 && decision.useWarmupPrefix
                ? min(layout.warmupPrefixSamples, decision.start) : 0
            let visibleChunkSamples = max(
                ASRConstants.samplesPerEncoderFrame,
                layout.chunkSamples - defaultWarmupSamples
            )
            guard available > decision.start + visibleChunkSamples else { return }

            // Only the last window consults `speechEnd`, and this one is not it.
            guard
                let plan = planWindow(for: decision, in: chunker, speechEnd: available),
                !plan.isLastChunk
            else { return }

            try await run(plan, in: chunker)
        }
    }

    private func planWindow(
        for decision: ChunkProcessor.ChunkStartDecision,
        in chunker: ChunkProcessor,
        speechEnd: Int
    ) -> ChunkProcessor.WindowPlan? {
        chunker.planWindow(
            chunkIndex: chunkIndex,
            chunkStart: decision.start,
            useWarmupPrefix: decision.useWarmupPrefix,
            warmupPrefixSamples: layout.warmupPrefixSamples,
            chunkSamples: layout.chunkSamples,
            melContextSamples: layout.melContextSamples,
            endAligned: endAligned,
            speechEnd: speechEnd
        )
    }

    private func run(_ plan: ChunkProcessor.WindowPlan, in chunker: ChunkProcessor) async throws {
        let samples = try chunker.readSamples(
            offset: plan.contextStart,
            count: plan.audioEnd - plan.contextStart
        )
        let started = Date()

        // `process`'s task group body for one window. The fresh decoder state
        // is what makes a window depend only on its own samples.
        var decoderState = TdtDecoderState.make(decoderLayers: decoderLayers)
        decoderState.reset()
        let (tokens, timestamps, confidences, durations) = try await ChunkProcessor.transcribeChunk(
            samples: samples,
            contextSamples: plan.contextSamples,
            chunkStart: plan.chunkStartOffset,
            isLastChunk: plan.isLastChunk,
            using: manager,
            decoderState: &decoderState,
            maxModelSamples: ASRConstants.maxModelSamples,
            language: language,
            emitTokensAfterFrame: plan.emitTokensAfterFrame,
            initialTimeIndexOverride: plan.emitTokensAfterFrame == nil ? nil : 0
        )
        guard tokens.count == timestamps.count, tokens.count == confidences.count else {
            throw ASRError.processingFailed("Token, timestamp, and confidence arrays are misaligned")
        }
        let tokenDurations =
            durations.count == tokens.count ? durations : Array(repeating: 0, count: tokens.count)

        accumulatedProcessingTime += Date().timeIntervalSince(started)
        guard !isCancelled else { return }

        windows.append(
            zip(zip(zip(tokens, timestamps), confidences), tokenDurations).map {
                (token: $0.0.0.0, timestamp: $0.0.0.1, confidence: $0.0.1, duration: $0.1)
            }
        )
        previousStart = plan.chunkStart
        chunkIndex += 1
        pendingDecision = nil
    }

    // MARK: - Where the next window starts

    /// The start `process` would use for window `chunkIndex`, or nil while the
    /// audio that arrived so far cannot settle it: `process` takes the start
    /// from its precomputed list when the silence loop reached this index —
    /// exactly when `chunkIndex * strideSamples < totalSamples` — and otherwise
    /// advances the previous start by one stride. Only the final total decides
    /// between those, so until the recording is complete (`isFinal`) the
    /// undecidable case yields nil, which costs nothing: such a window cannot
    /// run yet anyway.
    private func nextStart(
        in chunker: ChunkProcessor,
        isFinal: Bool
    ) throws -> ChunkProcessor.ChunkStartDecision? {
        if let pendingDecision {
            return pendingDecision
        }
        if chunkIndex == 0 {
            return remember(ChunkProcessor.ChunkStartDecision(start: 0, useWarmupPrefix: false))
        }
        let target = chunkIndex * layout.strideSamples
        guard target < chunker.totalSamples else {
            guard isFinal else { return nil }
            return remember(
                ChunkProcessor.ChunkStartDecision(
                    start: previousStart + layout.strideSamples,
                    useWarmupPrefix: false
                )
            )
        }
        guard preferSilenceAlignment || layout.warmupPrefixSamples > 0 else {
            // `regularChunkStarts` is just the nominal grid.
            return remember(ChunkProcessor.ChunkStartDecision(start: target, useWarmupPrefix: false))
        }
        // The silence search reads forward of `target`, as far as the latest
        // start that still overlaps the previous window, plus one frame for the
        // energy window around the candidate. Deciding before that much audio
        // exists would search a truncated signal and settle on a start `process`
        // would not have chosen, and `remember` then freezes it. Waiting costs
        // nothing: the window cannot run until a further `chunkSamples` arrives.
        let searchLimit =
            previousStart + layout.chunkSamples - 5 * ASRConstants.samplesPerEncoderFrame
        guard isFinal || chunker.totalSamples >= searchLimit else { return nil }
        return remember(
            try chunker.nextSilenceAlignedStart(
                after: previousStart,
                target: target,
                chunkSamples: layout.chunkSamples,
                strideSamples: layout.strideSamples,
                canUseWarmupPrefix: layout.warmupPrefixSamples > 0
            )
        )
    }

    private func remember(
        _ decision: ChunkProcessor.ChunkStartDecision
    ) -> ChunkProcessor.ChunkStartDecision {
        pendingDecision = decision
        return decision
    }
}
