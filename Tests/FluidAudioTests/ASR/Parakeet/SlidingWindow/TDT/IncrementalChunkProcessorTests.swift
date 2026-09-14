#if os(macOS)
import Foundation
import XCTest

@testable import FluidAudio

/// The contract `IncrementalChunkProcessor` exists to keep: audio fed a second
/// at a time must transcribe to exactly what a batch run produces from the same
/// samples.
///
/// Needs the v3 models in the local cache and `say` to render the audio, so it
/// skips rather than downloads. Populate the cache with any FluidAudio ASR run,
/// or `swift run fluidaudio transcribe <file>`.
final class IncrementalChunkProcessorTests: XCTestCase {

    /// Long enough for three windows, so the silence-aligned starts and the
    /// seam merge are exercised rather than a single padded pass.
    private static let longText = """
        The morning train left the station a few minutes after seven.
        A thin fog still covered the fields on both sides of the track.
        Most of the passengers were reading or looking out of the window.
        The conductor walked slowly through the carriage and checked every ticket.
        By the time we reached the river the sun had burned the fog away.
        The old bridge has been closed to cars for almost a decade.
        People still cross it on foot to reach the market on the far bank.
        On Saturdays the square fills with stalls selling bread, cheese and flowers.
        My grandmother used to buy her apples from the same family every week.
        She said the trick was to press the skin gently near the stem.
        If it gives a little, the fruit is ripe and will taste sweet.
        The bakery on the corner opens before dawn and closes at noon.
        Their rye loaf is dense and dark and keeps well for several days.
        Behind the church there is a small park with a pond and two benches.
        """

    /// A few seconds, comfortably inside the model's fifteen-second window.
    private static let shortText = """
        The morning train left the station a few minutes after seven.
        A thin fog still covered the fields on both sides of the track.
        """

    func testIncrementalMatchesBatchTranscription() async throws {
        let (samples, manager) = try await loadFixture(speaking: Self.longText)
        XCTAssertGreaterThan(samples.count, 2 * ASRConstants.maxModelSamples)

        let batch = try await ChunkProcessor(audioSamples: samples)
            .process(using: manager, startTime: Date())
        let incremental = try await transcribeIncrementally(samples, using: manager)

        XCTAssertFalse(batch.text.isEmpty, "the batch run produced no text to compare against")
        XCTAssertEqual(
            incremental.text, batch.text,
            "incremental text diverged from the batch transcription"
        )
        XCTAssertEqual(
            incremental.tokenTimings?.map(\.tokenId), batch.tokenTimings?.map(\.tokenId),
            "incremental token ids diverged from the batch transcription"
        )
        XCTAssertEqual(
            incremental.tokenTimings?.map(\.startTime), batch.tokenTimings?.map(\.startTime),
            "incremental token timings diverged from the batch transcription"
        )
    }

    /// Audio that fits the model's window never reaches `ChunkProcessor` in a
    /// batch run, so the contract at this length is against `AsrManager` itself.
    /// Comparing with `ChunkProcessor.process` instead would pass while the
    /// texts a caller actually sees differ.
    func testIncrementalMatchesBatchTranscriptionForShortAudio() async throws {
        let (samples, manager) = try await loadFixture(speaking: Self.shortText)
        XCTAssertLessThanOrEqual(samples.count, ASRConstants.maxModelSamples)

        var batchState = TdtDecoderState.make(decoderLayers: await manager.decoderLayerCount)
        let batch = try await manager.transcribe(samples, decoderState: &batchState)
        let incremental = try await transcribeIncrementally(samples, using: manager)

        XCTAssertFalse(batch.text.isEmpty, "the batch run produced no text to compare against")
        XCTAssertEqual(
            incremental.text, batch.text,
            "incremental text diverged from the batch transcription of short audio"
        )
    }

    // MARK: - Fixture

    /// Renders `text` with `say` and loads the v3 models, skipping when either
    /// is unavailable.
    private func loadFixture(speaking text: String) async throws -> ([Float], AsrManager) {
        try XCTSkipIf(
            ProcessInfo.processInfo.environment["CI"] != nil,
            "Skipping model-dependent test in CI environment"
        )
        let modelDirectory = AsrModels.defaultCacheDirectory(for: .v3)
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: modelDirectory.path),
            "parakeet-tdt-0.6b-v3 is not in the local model cache"
        )
        guard let samples = try Self.synthesiseSpeech(text) else {
            throw XCTSkip("`say` is not available to render the test audio")
        }

        let manager = AsrManager()
        try await manager.loadModels(AsrModels.load(from: modelDirectory, version: .v3))
        addTeardownBlock { await manager.cleanup() }
        return (samples, manager)
    }

    /// Feeds `samples` a second at a time, as a live capture would.
    private func transcribeIncrementally(
        _ samples: [Float],
        using manager: AsrManager
    ) async throws -> ASRResult {
        let incremental = try await IncrementalChunkProcessor(manager: manager)
        for offset in stride(from: 0, to: samples.count, by: ASRConstants.sampleRate) {
            let end = min(offset + ASRConstants.sampleRate, samples.count)
            try await incremental.append(Array(samples[offset..<end]))
        }
        return try await incremental.finish()
    }

    /// Renders `text` with `say`, the only speech this repository can generate
    /// without a download. Nil when `say` is unavailable or fails.
    private static func synthesiseSpeech(_ text: String) throws -> [Float]? {
        let audioURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("IncrementalChunkProcessorTests-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: audioURL) }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/say")
        process.arguments = [
            "-v", "Samantha", "-r", "175",
            "-o", audioURL.path,
            "--file-format=WAVE", "--data-format=LEF32@16000",
            text,
        ]
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        do {
            try process.run()
        } catch {
            return nil
        }
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { return nil }

        return try AudioConverter().resampleAudioFile(audioURL)
    }
}
#endif
