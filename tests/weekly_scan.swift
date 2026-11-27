import CoreGraphics
import CoreText
import Foundation

var bounds = CGRect(x: 0, y: 0, width: 1000, height: 300)
let raster = CGContext(data: nil, width: 1000, height: 300, bitsPerComponent: 8,
                       bytesPerRow: 0, space: CGColorSpaceCreateDeviceGray(), bitmapInfo: 0)!
raster.setFillColor(gray: 1, alpha: 1)
raster.fill(bounds)
raster.textPosition = CGPoint(x: 40, y: 150)
let font = CTFontCreateWithName("Helvetica" as CFString, 32, nil)
let text = NSAttributedString(
    string: "The FableDesk scanner reference is ZEBRA 741.",
    attributes: [NSAttributedString.Key(kCTFontAttributeName as String): font])
CTLineDraw(CTLineCreateWithAttributedString(text), raster)
let url = URL(fileURLWithPath: CommandLine.arguments[1]) as CFURL
let pdf = CGContext(url, mediaBox: &bounds, nil)!
pdf.beginPDFPage(nil)
pdf.draw(raster.makeImage()!, in: bounds)
pdf.endPDFPage()
pdf.closePDF()
