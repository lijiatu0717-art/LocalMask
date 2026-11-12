// Apple Vision local OCR (macOS).
// Persistent JSON-lines interface; input pixels never leave this machine.
import Foundation
import Vision
import ImageIO

while let line = readLine() {
    do {
        let input = try JSONSerialization.jsonObject(with: Data(line.utf8)) as! [String: Any]
        let bytes = Data(base64Encoded: input["image"] as! String)!
        guard let source = CGImageSourceCreateWithData(bytes as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw NSError(domain: "OCR", code: 1)
        }
        if input["mode"] as? String == "faces" {
            let request = VNDetectFaceRectanglesRequest()
            try VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
            let hits: [[String: Any]] = (request.results ?? []).map { observation in
                let b = observation.boundingBox
                return ["term": "face", "text": "face", "confidence": observation.confidence,
                        "box": [b.minX, 1-b.maxY, b.width, b.height]]
            }
            let data = try JSONSerialization.data(withJSONObject: ["hits": hits, "texts": []])
            print(String(data: data, encoding: .utf8)!); fflush(stdout)
            continue
        }
        let terms = input["terms"] as? [String] ?? []
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.recognitionLanguages = terms.contains(where: { $0.unicodeScalars.contains(where: { $0.value > 127 }) }) ? ["zh-Hans", "en-US"] : ["en-US"]
        request.usesLanguageCorrection = false
        request.customWords = terms
        try VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
        var hits: [[String: Any]] = []
        var texts: [String] = []
        for observation in request.results ?? [] {
            guard let candidate = observation.topCandidates(1).first else { continue }
            texts.append(candidate.string)
            for term in terms where !term.isEmpty {
                var start = candidate.string.startIndex
                while start < candidate.string.endIndex,
                      let range = candidate.string.range(of: term, options: [.caseInsensitive, .diacriticInsensitive], range: start..<candidate.string.endIndex) {
                    let box = (try? candidate.boundingBox(for: range))?.boundingBox ?? observation.boundingBox
                    hits.append(["term": term, "text": candidate.string,
                                 "confidence": candidate.confidence,
                                 "box": [box.minX, 1 - box.maxY, box.width, box.height]])
                    start = range.upperBound
                }
            }
        }
        let result: [String: Any] = ["hits": hits, "texts": texts]
        let data = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
        print(String(data: data, encoding: .utf8)!)
        fflush(stdout)
    } catch {
        print("{\"error\":\"OCR decode or recognition failed\"}")
        fflush(stdout)
    }
}
