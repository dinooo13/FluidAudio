#if os(macOS)
import AVFoundation
import XCTest

@testable import FluidAudio

/// The contract `IncrementalChunkProcessor` exists to keep: audio fed a second
/// at a time must transcribe to exactly what `ChunkProcessor.process` produces
/// from the same samples in one go.
///
/// Needs the v3 models in the local cache and `say` to render the audio, so it
/// skips rather than downloads. Populate the cache with any FluidAudio ASR run,
/// or `swift run fluidaudio transcribe <file>`.
final class IncrementalChunkProcessorTests: XCTestCase {

    /// Long enough for three windows, so both paths exercise the silence-aligned
    /// starts and the seam merge rather than a single padded pass.
    private let spokenText = """
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

    func testIncrementalMatchesBatchTranscription() async throws {
        try XCTSkipIf(
            ProcessInfo.processInfo.environment["CI"] != nil,
            "Skipping model-dependent test in CI environment"
        )
        let modelDirectory = AsrModels.defaultCacheDirectory(for: .v3)
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: modelDirectory.path),
            "parakeet-tdt-0.6b-v3 is not in the local model cache"
        )
        guard let samples = try Self.synthesiseSpeech(spokenText) else {
            throw XCTSkip("`say` is not available to render the test audio")
        }
        // Three windows' worth: the whole point is the seam handling.
        XCTAssertGreaterThan(samples.count, 2 * ASRConstants.maxModelSamples)

        let models = try await AsrModels.load(from: modelDirectory, version: .v3)
        let manager = AsrManager()
        try await manager.loadModels(models)
        defer { Task { await manager.cleanup() } }

        let batch = ChunkProcessor(audioSamples: samples)
        let batchResult = try await batch.process(using: manager, startTime: Date())

        let incremental = try await IncrementalChunkProcessor(manager: manager)
        var offset = 0
        let piece = ASRConstants.sampleRate  // one second at a time
        while offset < samples.count {
            let end = min(offset + piece, samples.count)
            try await incremental.append(Array(samples[offset..<end]))
            offset = end
        }
        let incrementalResult = try await incremental.finish()

        XCTAssertFalse(batchResult.text.isEmpty, "the batch run produced no text to compare against")
        XCTAssertEqual(
            incrementalResult.text,
            batchResult.text,
            "incremental text diverged from the batch transcription"
        )
        XCTAssertEqual(
            incrementalResult.tokenTimings?.map(\.tokenId),
            batchResult.tokenTimings?.map(\.tokenId),
            "incremental token ids diverged from the batch transcription"
        )
        XCTAssertEqual(
            incrementalResult.tokenTimings?.map(\.startTime),
            batchResult.tokenTimings?.map(\.startTime),
            "incremental token timings diverged from the batch transcription"
        )
    }

    /// Audio that fits the model's window never reaches `ChunkProcessor` in a
    /// batch run, so the contract at this length is against `AsrManager`
    /// itself. Comparing with `ChunkProcessor.process` instead would pass
    /// while the texts a caller actually sees differ.
    func testIncrementalMatchesBatchTranscriptionForShortAudio() async throws {
        try XCTSkipIf(
            ProcessInfo.processInfo.environment["CI"] != nil,
            "Skipping model-dependent test in CI environment"
        )
        let modelDirectory = AsrModels.defaultCacheDirectory(for: .v3)
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: modelDirectory.path),
            "parakeet-tdt-0.6b-v3 is not in the local model cache"
        )
        guard let samples = try Self.synthesiseSpeech(Self.shortSpokenText) else {
            throw XCTSkip("`say` is not available to render the test audio")
        }
        // Under one model window: no window can run before the end.
        XCTAssertLessThanOrEqual(samples.count, ASRConstants.maxModelSamples)

        let models = try await AsrModels.load(from: modelDirectory, version: .v3)
        let manager = AsrManager()
        try await manager.loadModels(models)
        defer { Task { await manager.cleanup() } }

        var batchState = TdtDecoderState.make(decoderLayers: await manager.decoderLayerCount)
        let batchResult = try await manager.transcribe(samples, decoderState: &batchState)

        let incremental = try await IncrementalChunkProcessor(manager: manager)
        var offset = 0
        let piece = ASRConstants.sampleRate
        while offset < samples.count {
            let end = min(offset + piece, samples.count)
            try await incremental.append(Array(samples[offset..<end]))
            offset = end
        }
        let incrementalResult = try await incremental.finish()

        XCTAssertFalse(batchResult.text.isEmpty, "the batch run produced no text to compare against")
        XCTAssertEqual(
            incrementalResult.text,
            batchResult.text,
            "incremental text diverged from the batch transcription of short audio"
        )
    }

    /// A few seconds, comfortably inside the model's fifteen-second window.
    private static let shortSpokenText = """
        The morning train left the station a few minutes after seven.
        A thin fog still covered the fields on both sides of the track.
        """

    /// Renders the script with `say` straight to 16 kHz mono Float32 — the
    /// format the models take — and returns the samples. Nil when `say` fails.
    private static func synthesiseSpeech(_ text: String) throws -> [Float]? {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("IncrementalChunkProcessorTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let scriptURL = directory.appendingPathComponent("script.txt")
        let audioURL = directory.appendingPathComponent("speech.wav")
        try text.write(to: scriptURL, atomically: true, encoding: .utf8)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/say")
        process.arguments = [
            "-v", "Samantha", "-r", "175",
            "-f", scriptURL.path,
            "-o", audioURL.path,
            "--file-format=WAVE", "--data-format=LEF32@16000",
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

        let file = try AVAudioFile(forReading: audioURL)
        guard
            let buffer = AVAudioPCMBuffer(
                pcmFormat: file.processingFormat,
                frameCapacity: AVAudioFrameCount(file.length)
            )
        else { return nil }
        try file.read(into: buffer)
        guard let channel = buffer.floatChannelData?[0] else { return nil }
        return Array(UnsafeBufferPointer(start: channel, count: Int(buffer.frameLength)))
    }
}
#endif
