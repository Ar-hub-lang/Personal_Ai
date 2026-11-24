import AppKit
import Foundation
import PDFKit
import Vision

let arguments = Array(CommandLine.arguments.dropFirst())
guard let path = arguments.first,
      let document = PDFDocument(url: URL(fileURLWithPath: path)) else {
    throw CocoaError(.fileReadCorruptFile)
}

let texts = try arguments.dropFirst().map { value -> String in
    guard let number = Int(value), let page = document.page(at: number - 1) else {
        throw CocoaError(.fileReadCorruptFile)
    }
    let image = page.thumbnail(of: NSSize(width: 2000, height: 2000), for: .cropBox)
    guard let pixels = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        throw CocoaError(.fileReadCorruptFile)
    }
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    try VNImageRequestHandler(cgImage: pixels).perform([request])
    return (request.results ?? []).compactMap { $0.topCandidates(1).first?.string }.joined(separator: "\n")
}

let encoded = try JSONEncoder().encode(texts)
FileHandle.standardOutput.write(encoded)
