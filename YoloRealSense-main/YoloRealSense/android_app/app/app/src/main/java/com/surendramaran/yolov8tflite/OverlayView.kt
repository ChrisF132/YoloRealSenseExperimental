package com.surendramaran.yolov8tflite

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.AttributeSet
import android.util.Log
import android.view.View
import androidx.core.content.ContextCompat
import java.util.LinkedList
import kotlin.math.max

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results = listOf<BoundingBox>()
    private var boxPaint = Paint()
    private var textBackgroundPaint = Paint()

    private var bounds = Rect()
    // In your OverlayView class
    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 48f
        style = Paint.Style.FILL
        strokeWidth = 2f
        isAntiAlias = true
    }

    init {
        initPaints()
    }
    private val distancePaint = Paint().apply {
        color = Color.CYAN
        textSize = 40f
    }

    fun clear() {
        textPaint.reset()
        textBackgroundPaint.reset()
        boxPaint.reset()
        invalidate()
        initPaints()
    }

    private fun initPaints() {
        textBackgroundPaint.color = Color.BLACK
        textBackgroundPaint.style = Paint.Style.FILL
        textBackgroundPaint.textSize = 50f

        textPaint.color = Color.WHITE
        textPaint.style = Paint.Style.FILL
        textPaint.textSize = 50f

        boxPaint.color = ContextCompat.getColor(context!!, R.color.bounding_box_color)
        boxPaint.strokeWidth = 8F
        boxPaint.style = Paint.Style.STROKE
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        results.forEach { box ->
            // 1. Convert normalized coordinates to view coordinates
            val scaleX = width / 640f  // 1080 / 640
            val scaleY = height / 640f // 1440 / 640

            val left = box.x1 * scaleX
            val top = box.y1 * scaleY
            val right = box.x2 * scaleX
            val bottom = box.y2 * scaleY

            Log.d("BOX_DRAW", "Bitmap size: ${width} x ${height}")
            Log.d("BOX_DRAW", "Drawing box: (${box.x1}, ${box.y1}), (${box.x2}, ${box.y2})")


            // 2. Draw bounding box (keep your existing style)
            canvas.drawRect(left, top, right, bottom, boxPaint)

            // 3. Prepare combined label text
            val labelText = "${box.clsName} (${"%.1f".format(box.distance)}m)"

            // 4. Calculate text bounds for background
            textPaint.getTextBounds(labelText, 0, labelText.length, bounds)

            // 5. Draw text background
            canvas.drawRect(
                left,
                top,
                left + bounds.width() + BOUNDING_RECT_TEXT_PADDING,
                top + bounds.height() + BOUNDING_RECT_TEXT_PADDING,
                textBackgroundPaint
            )

            // 6. Draw class name + distance
            canvas.drawText(labelText, left, top + bounds.height(), textPaint)

            // 7. Optional: Draw distance at bottom right of box
            val distanceText = "${"%.1f m".format(box.distance)}"
            canvas.drawText(
                distanceText,
                right - distancePaint.measureText(distanceText) - 10f,
                bottom - 10f,
                distancePaint
            )
        }
    }

    fun setResults(boundingBoxes: List<BoundingBox>) {
        results = boundingBoxes
        invalidate()
    }

    companion object {
        private const val BOUNDING_RECT_TEXT_PADDING = 8
    }
}