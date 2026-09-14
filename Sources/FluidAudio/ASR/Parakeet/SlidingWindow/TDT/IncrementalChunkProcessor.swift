import Foundation

/// The utterance so far, behind the protocol `ChunkProcessor` reads audio
/// through. `sampleCount` is what has arrived, so a `ChunkProcessor` built on
/// it sees exactly the prefix that exists right now. Semantics match
/// `ArrayAudioSampleSource`, including leaving the tail of a read untouched
/// when it runs past the end.
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
        guard count > 0 else { return }
        guard !samples.isEmpty else { return }
        let clampedOffset = max(0, offset)
        guard clampedOffset < samples.count else { return }
        let available = min(samples.count - clampedOffset, count)
        samples.withUnsafeBufferPointer { pointer in
            destination.update(
                from: pointer.baseAddress!.advanced(by: clampedOffset),
                count: available
            )
        }
    }
}

/// `ChunkProcessor` for audio that arrives over time. Windows run as soon as
/// the audio they need exists; `finish()` runs the last window and the same
/// merge `ChunkProcessor.process` runs, so the text is identical to a batch
/// transcription of the same samples.
///
/// Why that identity holds: `ChunkProcessor` starts every window from a fresh
/// decoder state, so a window depends only on its own samples, and its start
/// rule for window *n+1* reads no audio past the end of window *n* (see
/// `nextSilenceAlignedStart`). Every window but the last can therefore be laid
/// out and run while the recording is still going, and the merge at the end
/// sees the same windows in the same order that a batch run would have
/// produced.
///
/// One window runs at a time — that is the point. The caller pays a single
/// window at release instead of one per fifteen seconds of speech.
public actor IncrementalChunkProcessor {

    private let manager: AsrManager
    private let language: Language?
    private let source = GrowingAudioSampleSource()

    private let decoderLayers: Int
    private let maxModelSamples: Int
    private let endAligned: Bool
    private let preferSilenceAlignment: Bool
    private let layout:
        (
            chunkSamples: Int, strideSamples: Int, melContextSamples: Int,
            warmupPrefixSamples: Int
        )

    /// Finished windows in order, ready for the shared merge.
    private var windows: [[ChunkProcessor.TokenWindow]] = []
    /// Index of the next window to lay out.
    private var chunkIndex = 0
    /// Start of the window laid out most recently.
    private var previousStart = 0
    /// Start of window `chunkIndex`, once enough audio exists to decide it.
    private var pendingStart: Int?
    private var pendingUseWarmupPrefix = false

    /// Encoder and decoder time across every window, so the merged result
    /// reports the compute the utterance cost rather than its wall clock.
    private var accumulatedProcessingTime: TimeInterval = 0

    private var isRunningWindow = false
    private var isCancelled = false
    private var isFinished = false

    public init(manager: AsrManager, language: Language? = nil) async throws {
        let melChunkContext = await manager.melChunkContext
        let modelVersion = await manager.modelVersion
        let dualDecodeArbitration = await manager.dualDecodeArbitration

        // `process` hands v3/no-mel over to a separate two-decode path with
        // its own window bookkeeping; there is nothing here to share with it.
        if dualDecodeArbitration, !melChunkContext, modelVersion == .v3 {
            throw ASRError.processingFailed(
                "IncrementalChunkProcessor does not support dualDecodeArbitration"
            )
        }

        self.manager = manager
        self.language = language
        self.decoderLayers = await manager.decoderLayerCount
        self.maxModelSamples = ASRConstants.maxModelSamples
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
        // The final window must end at the last speech-bearing frame, not EOF.
        let speechEnd = endAligned ? try chunker.speechEndSamples() : totalSamples

        // `process` keeps looping until a window reports itself last. Normally
        // exactly one window is left here, but a start that could not be
        // decided while the audio was still arriving can leave two.
        while !isCancelled {
            guard let decision = try startForFinalWindows(in: chunker, totalSamples: totalSamples),
                decision.start < totalSamples
            else { break }
            guard
                let plan = chunker.planWindow(
                    chunkIndex: chunkIndex,
                    chunkStart: decision.start,
                    useWarmupPrefix: decision.useWarmupPrefix,
                    warmupPrefixSamples: layout.warmupPrefixSamples,
                    chunkSamples: layout.chunkSamples,
                    melContextSamples: layout.melContextSamples,
                    endAligned: endAligned,
                    speechEnd: speechEnd
                )
            else { break }

            try await run(plan, in: chunker)
            if plan.isLastChunk { break }
        }

        guard !isCancelled else { throw CancellationError() }

        let elapsed = Date().timeIntervalSince(finishStarted)
        return try await chunker.finalize(
            chunkOutputs: windows,
            using: manager,
            repairWorker: manager,
            decoderLayers: decoderLayers,
            maxModelSamples: maxModelSamples,
            language: language,
            // `finalize` reports `Date().timeIntervalSince(startTime)`; back-date
            // it so the figure is the compute every window cost, not the
            // seconds the speaker took.
            startTime: Date().addingTimeInterval(-(accumulatedProcessingTime + elapsed))
        )
    }

    public func cancel() {
        isCancelled = true
        windows.removeAll()
        pendingStart = nil
        source.removeAll()
    }

    // MARK: - Window scheduling

    private func runReadyWindows() async throws {
        guard !isRunningWindow else { return }
        isRunningWindow = true
        defer { isRunningWindow = false }

        while !isCancelled, !isFinished {
            let chunker = ChunkProcessor(sampleSource: source)
            let available = chunker.totalSamples
            guard let decision = try startForNextWindow(in: chunker, available: available) else {
                return
            }

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

            guard
                let plan = chunker.planWindow(
                    chunkIndex: chunkIndex,
                    chunkStart: decision.start,
                    useWarmupPrefix: decision.useWarmupPrefix,
                    warmupPrefixSamples: layout.warmupPrefixSamples,
                    chunkSamples: layout.chunkSamples,
                    melContextSamples: layout.melContextSamples,
                    // Only the last window consults `speechEnd`, and this one
                    // is not it.
                    endAligned: endAligned,
                    speechEnd: available
                ), !plan.isLastChunk
            else { return }

            try await run(plan, in: chunker)
        }
    }

    private func run(_ plan: ChunkProcessor.WindowPlan, in chunker: ChunkProcessor) async throws {
        let samples = try chunker.readSamples(
            offset: plan.contextStart,
            count: plan.audioEnd - plan.contextStart
        )
        let started = Date()
        let tokens = try await ChunkProcessor.runWindow(
            samples: samples,
            plan: plan,
            using: manager,
            decoderLayers: decoderLayers,
            maxModelSamples: maxModelSamples,
            language: language
        )
        accumulatedProcessingTime += Date().timeIntervalSince(started)
        guard !isCancelled else { return }

        windows.append(tokens)
        previousStart = plan.chunkStart
        chunkIndex += 1
        pendingStart = nil
    }

    // MARK: - Where the next window starts

    /// The start `process` would use for window `chunkIndex`, or nil while the
    /// audio that arrived so far cannot settle it.
    ///
    /// `process` takes the start from the precomputed list when the silence
    /// loop reached this index, which it did exactly when
    /// `chunkIndex * strideSamples < totalSamples`, and otherwise advances the
    /// previous start by one stride. Only the final total decides between
    /// those, so hold off until `available` has passed the nominal target.
    /// That costs nothing: a window whose start lies at or beyond `available`
    /// cannot run yet anyway, and neither can one within a stride of it.
    private func startForNextWindow(
        in chunker: ChunkProcessor,
        available: Int
    ) throws -> ChunkProcessor.ChunkStartDecision? {
        if let pendingStart {
            return ChunkProcessor.ChunkStartDecision(
                start: pendingStart,
                useWarmupPrefix: pendingUseWarmupPrefix
            )
        }
        if chunkIndex == 0 {
            return remember(ChunkProcessor.ChunkStartDecision(start: 0, useWarmupPrefix: false))
        }
        let target = chunkIndex * layout.strideSamples
        guard available > target else { return nil }
        return remember(try silenceOrRegularStart(in: chunker, target: target))
    }

    /// The same decision once the recording is complete, where the fallback
    /// branch `process` takes past the end of its start list is reachable.
    private func startForFinalWindows(
        in chunker: ChunkProcessor,
        totalSamples: Int
    ) throws -> ChunkProcessor.ChunkStartDecision? {
        if let pendingStart {
            return ChunkProcessor.ChunkStartDecision(
                start: pendingStart,
                useWarmupPrefix: pendingUseWarmupPrefix
            )
        }
        if chunkIndex == 0 {
            return remember(ChunkProcessor.ChunkStartDecision(start: 0, useWarmupPrefix: false))
        }
        let target = chunkIndex * layout.strideSamples
        guard target < totalSamples else {
            return remember(
                ChunkProcessor.ChunkStartDecision(
                    start: previousStart + layout.strideSamples,
                    useWarmupPrefix: false
                )
            )
        }
        return remember(try silenceOrRegularStart(in: chunker, target: target))
    }

    private func silenceOrRegularStart(
        in chunker: ChunkProcessor,
        target: Int
    ) throws -> ChunkProcessor.ChunkStartDecision {
        guard preferSilenceAlignment || layout.warmupPrefixSamples > 0 else {
            // `regularChunkStarts` is just the nominal grid.
            return ChunkProcessor.ChunkStartDecision(start: target, useWarmupPrefix: false)
        }
        return try chunker.nextSilenceAlignedStart(
            after: previousStart,
            target: target,
            chunkSamples: layout.chunkSamples,
            strideSamples: layout.strideSamples,
            canUseWarmupPrefix: layout.warmupPrefixSamples > 0
        )
    }

    private func remember(
        _ decision: ChunkProcessor.ChunkStartDecision
    ) -> ChunkProcessor.ChunkStartDecision {
        pendingStart = decision.start
        pendingUseWarmupPrefix = decision.useWarmupPrefix
        return decision
    }
}
