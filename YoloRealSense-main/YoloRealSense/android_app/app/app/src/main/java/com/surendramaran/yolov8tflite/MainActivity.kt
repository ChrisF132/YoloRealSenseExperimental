package com.surendramaran.yolov8tflite



import android.Manifest
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.media.AudioAttributes
import android.media.AudioDeviceInfo
import android.media.AudioFocusRequest
import android.media.AudioManager
import android.media.MediaRecorder
import android.os.Bundle
import android.speech.RecognizerIntent
import android.speech.RecognitionListener
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.surendramaran.yolov8tflite.Constants.LABELS_PATH
import com.surendramaran.yolov8tflite.Constants.MODEL_PATH
import com.surendramaran.yolov8tflite.databinding.ActivityMainBinding
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import android.os.Vibrator
import android.os.VibrationEffect
import android.content.Context
import android.content.IntentFilter
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
import android.os.Handler
import android.os.Looper
import com.intel.realsense.librealsense.Config
import com.intel.realsense.librealsense.DepthFrame
import com.intel.realsense.librealsense.DeviceListener
import com.intel.realsense.librealsense.Extension
import com.intel.realsense.librealsense.FrameSet
import com.intel.realsense.librealsense.Pipeline
import com.intel.realsense.librealsense.RsContext
import com.intel.realsense.librealsense.StreamFormat
import com.intel.realsense.librealsense.StreamType
import com.intel.realsense.librealsense.UsbUtilities.ACTION_USB_PERMISSION
import com.intel.realsense.librealsense.VideoFrame
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizer
import com.google.mlkit.vision.text.latin.TextRecognizerOptions

import java.nio.ByteBuffer

enum class DetectionMode {
    START_DETECTION, EXPLAIN_SURROUNDING
}

class MainActivity : AppCompatActivity(), Detector.DetectorListener {

    private var testvalue = true
    private lateinit var binding: ActivityMainBinding
    private lateinit var detector: Detector
    private var isDetectionActive = false
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var textToSpeech: TextToSpeech
    private var isTTSInitialized = false
    private lateinit var speechRecognizer: SpeechRecognizer
    private lateinit var speechRecognizerIntent: Intent
    private lateinit var vibrator: Vibrator
    private var currentMode: DetectionMode = DetectionMode.START_DETECTION
    private var isProcessingCommand = false
    private val trackedObjects = mutableMapOf<String, Long>()
    private var lastAnnouncementTime = 0L
    private val announcementCooldown = 2000L
    private val minConfidence = 0.40f
    private lateinit var pipeline: Pipeline
    private var streamingThread: Thread? = null
    private var currentDepthFrame: DepthFrame? = null
    private lateinit var rsContext: RsContext
    private var isPipelineRunning = false
    private var isUsbConnected = false
    private var errorBackoffDelay = 1000L
    private val maxBackoffDelay = 10000L
    private lateinit var audioManager: AudioManager
    private lateinit var audioFocusRequest: AudioFocusRequest
    private lateinit var speechListener: RecognitionListener
    private var isListening = false
    private var usbReady = false
    private var isReadingText = false
    private lateinit var textRecognizer: TextRecognizer
    private var lastColorBitmap: Bitmap? = null

    //For single frame testing
    private var hasSavedSingleFrame = false
    private var isDetectorBusy = false




    private val usbReceiver = object : BroadcastReceiver() {

        override fun onReceive(context: Context, intent: Intent) {
            when (intent.action) {
                ACTION_USB_PERMISSION -> {

                    val device = intent.getParcelableExtra<UsbDevice>(UsbManager.EXTRA_DEVICE)
                    if (intent.getBooleanExtra(UsbManager.EXTRA_PERMISSION_GRANTED, false)) {

                        Log.d(TAG, "USB permission granted for ${device?.deviceName}")
                        isUsbConnected = true

                        configureAudioRouting()
                        startRealsensePipeline()


                    } else {
                        Log.w(TAG, "USB permission denied for ${device?.deviceName}")
                    }
                }
                UsbManager.ACTION_USB_DEVICE_ATTACHED -> {
                    Log.d(TAG, "USB attached")

                    isUsbConnected = true
                    configureAudioRouting()
                    pauseSpeechRecognition()


                    val count = try {
                        rsContext.queryDevices().deviceCount
                    } catch (e: Exception) {
                        Log.e(TAG, "Error querying RealSense devices in USB attach", e)
                        0
                    }

                    Log.d(TAG, "Device count at attach time: $count")

                    if (count > 0) {
                        Log.d(TAG, "Device present, checking permissions")
                        checkUsbPermissions()
                    } else {
                        Log.w(TAG, "No devices found at USB attach time — waiting 500ms")
                        Handler(Looper.getMainLooper()).postDelayed({
                            val retryCount = rsContext.queryDevices().deviceCount
                            Log.d(TAG, "Device count after delay: $retryCount")
                            if (retryCount > 0) checkUsbPermissions()
                            else Log.e(TAG, "Still no RealSense devices found")
                        }, 500)
                    }
                }
                UsbManager.ACTION_USB_DEVICE_DETACHED -> {
                    Log.d(TAG, "USB disconnected - restoring audio")
                    isUsbConnected = false
                    resumeSpeechRecognition()
                }
            }
        }
    }

    private fun configureAudioRouting() {
        audioManager.mode = AudioManager.MODE_IN_COMMUNICATION
        audioManager.isBluetoothScoOn = false
        audioManager.isSpeakerphoneOn = true


        audioManager.setParameters("no_usb_audio=1")
        audioManager.mode = AudioManager.MODE_IN_COMMUNICATION
        audioManager.setParameters("no_usb_audio=1")

    }



    private fun pauseSpeechRecognition() {
        stopListening()
    }



    private fun resumeSpeechRecognition() {
        if (!isProcessingCommand && !isUsbConnected) {
            startListening()
        }
    }

    private fun checkUsbPermissions() {
        val usbManager = getSystemService(Context.USB_SERVICE) as UsbManager
        val deviceList = usbManager.deviceList.values.firstOrNull()

        Log.d(TAG, "USB device list size: ${usbManager.deviceList.size}")
        Log.d(TAG, "RsContext device count: ${rsContext.queryDevices().deviceCount}")

        deviceList?.let { device ->
            if (!usbManager.hasPermission(device)) {
                val permissionIntent = PendingIntent.getBroadcast(
                    this,
                    0,
                    Intent(ACTION_USB_PERMISSION),
                    PendingIntent.FLAG_IMMUTABLE
                )
                usbManager.requestPermission(device, permissionIntent)

            } else {
                Log.d(TAG, "USB permission already granted for ${device.deviceName}")
                isUsbConnected = true
                configureAudioRouting()
                startRealsensePipeline()
            }
        }
    }



    private fun startRealsensePipeline(): Boolean {
        return try {
            pipeline = Pipeline().apply {
                start(Config().apply {
                    enableStream(StreamType.COLOR, 640, 480, StreamFormat.RGBA8)
                    enableStream(StreamType.DEPTH, 640, 480, StreamFormat.Z16)
                })
                Log.d(TAG, "Pipeline started with profile")
            }

            streamingThread = Thread {
                while (!Thread.interrupted()) {
                    try {
                        pipeline.waitForFrames(5000)?.use { frames ->
                            processFrame(frames)
                        }
                    } catch (e: RuntimeException) {
                        if (e.message?.contains("timeout") == true) {
                            Log.w(TAG, "Frame timeout - retrying...")
                        } else {
                            Log.e(TAG, "Frame processing error", e)
                        }
                    }
                    catch (e: InterruptedException) {
                        Log.e("YOLO", "Realsense thread interrupted during sleep")
                        break
                    }
                }
            }.apply { start() }

            isPipelineRunning = true
            true

        } catch (e: Exception) {
            Log.e(TAG, "Pipeline startup failed", e)
            false
        }
    }

    private fun stopRealsensePipeline() {
        streamingThread?.let { thread ->
            thread.interrupt()
            try {
                thread.join(1000)

            } catch (e: InterruptedException) {
                Log.w(TAG, "Thread interruption warning", e)
            }
        }

        streamingThread = null

        if (::pipeline.isInitialized && isPipelineRunning) {
            try {
                pipeline.stop()
                isPipelineRunning = false
                Log.d(TAG, "Pipeline stopped successfully")

            } catch (e: Exception) {
                Log.e(TAG, "Pipeline stop failed", e)
            }
        }
    }

    private fun processFrame(frames: FrameSet) {
        frames.first(StreamType.COLOR)?.use { colorFrame ->
            if (colorFrame.`is`(Extension.VIDEO_FRAME)) {
                val videoFrame = colorFrame.`as`<VideoFrame>(Extension.VIDEO_FRAME)
                val bitmap = videoFrame.toBitmap()
                val bitmapScaled = Bitmap.createScaledBitmap(bitmap, 640, 640, false)

                frames.first(StreamType.DEPTH)?.use { depthFrame ->
                    if (depthFrame.`is`(Extension.DEPTH_FRAME)) {
                        currentDepthFrame = depthFrame.`as`(Extension.DEPTH_FRAME)
                    }
                }

                Log.d(TAG, "Bitmap to detect size: ${bitmap.width}x${bitmap.height}")
                Log.d(TAG, "Bitmap recycled: ${bitmap.isRecycled}")
                val pixel = bitmap.getPixel(bitmap.width / 2, bitmap.height / 2)
                Log.d(TAG, "Center pixel ARGB: ${Integer.toHexString(pixel)}")

                lastColorBitmap = bitmap

                runOnUiThread {
                    binding.cameraPreview.setImageBitmap(bitmapScaled)
                    binding.inferenceTime.text = ""
                }

                //Run detection in the background
                if (isDetectionActive && ::detector.isInitialized && !isDetectorBusy) {
                    isDetectorBusy = true
                    cameraExecutor.execute {
                        val startTime = System.currentTimeMillis()
                        detector.detect(bitmapScaled)
                        val endTime = System.currentTimeMillis()
                        Log.d(TAG, "Frame inference took ${endTime - startTime}ms")
                        isDetectorBusy = false
                    }
                }
            }
        }
    }

    private fun VideoFrame.toBitmap(): Bitmap {
        return try {
            val byteArray = ByteArray(dataSize)
            getData(byteArray)
            val buffer = ByteBuffer.wrap(byteArray)

            val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            bmp.copyPixelsFromBuffer(buffer)

            bmp
        } catch (e: Exception) {
            Log.e(TAG, "Bitmap conversion failed", e)
            Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888)
        }
    }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        textRecognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
        val filter = IntentFilter().apply {

            addAction(ACTION_USB_PERMISSION)
            addAction(UsbManager.ACTION_USB_DEVICE_ATTACHED)
            addAction(UsbManager.ACTION_USB_DEVICE_DETACHED)
        }

        registerReceiver(usbReceiver, filter, Context.RECEIVER_NOT_EXPORTED)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        audioManager = getSystemService(AUDIO_SERVICE) as AudioManager

        audioFocusRequest = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT)
            .setAudioAttributes(AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_ASSISTANCE_ACCESSIBILITY)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                .build())
            .build()



        vibrator = getSystemService(Context.VIBRATOR_SERVICE) as Vibrator

        detector = Detector(baseContext, MODEL_PATH, LABELS_PATH, this)

        detector.setup()


        RsContext.init(applicationContext)

        rsContext = RsContext().apply {
            setDevicesChangedCallback(object : DeviceListener {
                override fun onDeviceAttach() {
                    Log.d(TAG, "RealSense device connected (onCreate)")
                    isUsbConnected = true
                }

                override fun onDeviceDetach() {
                    Log.d(TAG, "RealSense device disconnected (onCreate)")
                    isUsbConnected = false
                }
            })
        }

        Log.d(TAG, "Initial RealSense device count: ${rsContext.queryDevices().deviceCount}")

        // Initialize TTS

        Handler(Looper.getMainLooper()).postDelayed({
            textToSpeech = TextToSpeech(this) { status ->

                if (status == TextToSpeech.SUCCESS) {

                    val result = textToSpeech.setLanguage(Locale.US)

                    isTTSInitialized = result != TextToSpeech.LANG_MISSING_DATA &&

                            result != TextToSpeech.LANG_NOT_SUPPORTED

                    if (!isTTSInitialized) {

                        Log.e(TAG, "TTS Language not supported")

                    }

                } else {

                    Log.e(TAG, "TTS Initialization failed")

                }

            }
        }, 1000)



        // Setup Speech Recognition

        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this)

        speechRecognizerIntent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {

            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)

            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())

            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)

            putExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE, MediaRecorder.AudioSource.MIC)


        }



        speechListener = object : RecognitionListener {
            override fun onResults(results: Bundle?) {
                val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)

                matches?.let {

                    if (it.isEmpty()) return@let

                    val spokenText = it[0].lowercase(Locale.ROOT)

                    if (!isProcessingCommand) {
                        isProcessingCommand = true

                        when {

                            "start detection" in spokenText && !isDetectionActive -> {
                                if (allPermissionsGranted()) {
                                    speak("Starting detection now")
                                    isDetectionActive = true
                                    currentMode = DetectionMode.START_DETECTION
                                    trackedObjects.clear()
                                    isReadingText = false

                                    val connected = try {
                                        rsContext.queryDevices().deviceCount > 0
                                    } catch (e: Exception) {
                                        Log.e(TAG, "Failed to query RealSense devices", e)
                                        false
                                    }

                                    if (connected) {
                                        Log.d(TAG, "RealSense device detected — attempting to check permissions")
                                        checkUsbPermissions()
                                    } else {
                                        speak("RealSense camera is not connected")
                                        Log.w(TAG, "RealSense not connected according to rsContext")
                                    }
                                }
                            }

                            "stop detection" in spokenText && isDetectionActive -> {
                                speak("Stopping detection")
                                stopCamera()
                                isDetectionActive = false
                                isReadingText = false
                            }
                            "explain" in spokenText -> {
                                speak("Explaining the surroundings")
                                currentMode = DetectionMode.EXPLAIN_SURROUNDING
                                isReadingText = false
                            }

                            "read text" in spokenText -> {
                                if (isDetectionActive) {
                                    speak("Reading text")
                                    isReadingText = true
                                    lastColorBitmap?.let { bitmap ->
                                        processImageForText(bitmap)
                                    }
                                } else {
                                    speak("Please start detection first")
                                }
                            }
                        }
                        val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                        Log.d(TAG, "Recognized: ${matches?.joinToString()}")

                        Handler(Looper.getMainLooper()).postDelayed({

                            isProcessingCommand = false
                            safeStartListening()

                        }, 1500)
                    }
                }
                errorBackoffDelay = 1000L
            }
            override fun onError(error: Int) {
                isListening = false
                Log.e(TAG, "Speech recognition error: $error")

                when (error) {

                    SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> {
                        Log.e(TAG, "Missing permissions")
                        requestAudioPermission()
                    }
                    SpeechRecognizer.ERROR_CLIENT -> {
                        resetRecognizer()
                    }
                    SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> {
                        scheduleRetryWithBackoff()
                    }
                    SpeechRecognizer.ERROR_NO_MATCH -> {
                        Log.w(TAG, "No match - resuming listening")
                        scheduleRetryWithBackoff()
                    }
                    else -> {
                        scheduleRetryWithBackoff()
                    }
                }
            }
            override fun onReadyForSpeech(params: Bundle?) {}
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() {
                Log.d(TAG, "End of speech detected")
                isListening = false
                if (!isProcessingCommand) {
                    scheduleRetryWithBackoff()
                }
            }
            override fun onPartialResults(partialResults: Bundle?) {}
            override fun onEvent(eventType: Int, params: Bundle?) {}
        }
        speechRecognizer.setRecognitionListener(speechListener)

        if (!allPermissionsGranted()) {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        } else {
            resetRecognizer()
        }
        cameraExecutor = Executors.newSingleThreadExecutor()

    }

    private fun processImageForText(bitmap: Bitmap) {
        Log.d(TAG, "reached processImage")
        val image = InputImage.fromBitmap(bitmap, 0)
        textRecognizer.process(image)
            .addOnSuccessListener { visionText ->
                val textBlocks = visionText.textBlocks
                if (textBlocks.isNotEmpty()) {
                    val fullText = StringBuilder()
                    for (block in textBlocks) {
                        fullText.append(block.text).append("\n")
                    }
                    val textToRead = fullText.toString().trim()
                    if (textToRead.isNotEmpty()) {
                        speak("I found text: $textToRead")

                    } else {
                        speak("No readable text found")
                    }
                } else {
                    speak("No text detected")
                }
                isReadingText = false
            }
            .addOnFailureListener { e ->
                Log.e(TAG, "Text recognition failed", e)
                speak("Failed to read text")
                isReadingText = false
            }
    }

    private fun safeStartListening() {
        if (!isProcessingCommand && acquireAudioFocus()) {
            try {
                speechRecognizer.startListening(speechRecognizerIntent)
            } catch (e: SecurityException) {
                Log.e(TAG, "Microphone access denied", e)
                requestAudioPermission()
            } catch (e: IllegalStateException) {
                Log.e(TAG, "Recognizer in bad state", e)
                resetRecognizer()
            }
        }
    }




    private fun acquireAudioFocus(): Boolean {
        return audioManager.requestAudioFocus(audioFocusRequest) ==
                AudioManager.AUDIOFOCUS_REQUEST_GRANTED
    }


    private fun resetRecognizer() {
        stopListening()
        isListening = false

        try {
            speechRecognizer.destroy()
        } catch (e: Exception) {
            Log.w(TAG, "Failed to destroy speech recognizer", e)
        }

        Handler(Looper.getMainLooper()).postDelayed({
            if (SpeechRecognizer.isRecognitionAvailable(this)) {
                speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this)
                speechRecognizer.setRecognitionListener(speechListener)
                Log.d(TAG, "SpeechRecognizer reset complete")
                Handler(Looper.getMainLooper()).postDelayed({
                    safeStartListening()
                }, 300)
            } else {
                Log.e(TAG, "Speech recognition not available on device")
            }
        }, 300)
    }


    private fun requestAudioPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.RECORD_AUDIO),
                REQUEST_CODE_AUDIO_PERMISSION
            )
        }
    }


    private fun scheduleRetryWithBackoff() {
        Handler(Looper.getMainLooper()).postDelayed({
            if (!isProcessingCommand) {
                safeStartListening()
            }
        }, errorBackoffDelay)
        errorBackoffDelay = (errorBackoffDelay * 2).coerceAtMost(maxBackoffDelay)
    }



    private fun speak(text: String) {

        if (isTTSInitialized) {

            textToSpeech.speak(text, TextToSpeech.QUEUE_FLUSH, null, null)

        }

    }



    private fun startListening() {
        if (!isProcessingCommand && !isListening) {
            try {
                speechRecognizer.startListening(speechRecognizerIntent)
                isListening = true
                Log.d(TAG, "SpeechRecognizer: started listening")
            } catch (e: Exception) {
                Log.e(TAG, "Speech recognition start failed", e)
                isListening = false
            }
        }
    }



    private fun stopListening() {
        if (isListening) {
            try {
                speechRecognizer.stopListening()
                isListening = false
                Log.d(TAG, "SpeechRecognizer: stopped listening")
            } catch (e: Exception) {
                Log.e(TAG, "Speech recognition stop failed", e)
            }
        }
    }



    private fun startCamera() = startRealsensePipeline()

    private fun stopCamera() = stopRealsensePipeline()



    override fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long) {


        Log.d(TAG, "Overlay size: ${binding.overlay.width}x${binding.overlay.height}")
        Log.d(TAG, "BoundingBox sample: ${boundingBoxes.firstOrNull()}")
        Log.d(TAG, "YOLO DETECTION TRIGGERED: ${boundingBoxes.size} boxes")

        runOnUiThread {
            Log.d("DrawCheck", "Bitmap dimensions before drawing: ${binding.cameraPreview.drawable.intrinsicWidth} x ${binding.cameraPreview.drawable.intrinsicHeight}")
            binding.inferenceTime.text = "${inferenceTime}ms"

            binding.overlay.apply {

                setResults(boundingBoxes)

                invalidate()

            }

            val bitmap = binding.viewFinder.bitmap

            if (isReadingText && bitmap != null) {
                processImageForText(bitmap)
                return@runOnUiThread
            }


            if (!isTTSInitialized || !isDetectionActive) return@runOnUiThread



            val currentTime = System.currentTimeMillis()

            val highConfidenceBoxes = boundingBoxes.filter { it.confidence > minConfidence }



            highConfidenceBoxes.forEach { box ->
                val tensorInputWidth = 640f
                val tensorInputHeight = 640f
                val frameWidth = (currentDepthFrame?.width ?: 640).toFloat()
                val frameHeight = (currentDepthFrame?.height ?: 480).toFloat()

                val scaleX = frameWidth / tensorInputWidth
                val scaleY = frameHeight / tensorInputHeight

                val dfWidth = currentDepthFrame?.width ?: 0
                val dfHeight = currentDepthFrame?.height ?: 0

                if(dfWidth <= 0 || dfHeight <= 0) {
                    Log.e("YOLO", "Invalid dimensions, width = $dfWidth height = $dfHeight")
                    return@runOnUiThread
                }

                val realX = (box.cx * (currentDepthFrame?.width ?: 640)).toInt().coerceIn(0, (currentDepthFrame?.width ?: 640) - 1)
                val realY = (box.cy * (currentDepthFrame?.height ?: 480)).toInt().coerceIn(0, (currentDepthFrame?.height ?: 480) - 1)


                val distance = currentDepthFrame?.getDistance(realX, realY) ?: 0f
                box.distance = distance

                Log.d(TAG, "${box.clsName} (${box.confidence}) at ${"%.2f".format(distance)}m")
            }





            if (highConfidenceBoxes.isNotEmpty() &&

                !textToSpeech.isSpeaking &&

                currentTime - lastAnnouncementTime >= announcementCooldown) {



                val message = buildString {

                    append("Detected ")

                    highConfidenceBoxes.forEachIndexed { index, box ->

                        append("${box.clsName} at ${"%.1f".format(box.distance)} meters")

                        if (index < highConfidenceBoxes.size - 1) append(", ")

                    }

                }



                safeVibrate()

                speak(message)

                lastAnnouncementTime = currentTime

            }

        }

    }



    override fun onEmptyDetect() {

        runOnUiThread {

            binding.overlay.invalidate()

        }

    }



    private fun safeVibrate() {

        try {

            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {

                vibrator.vibrate(VibrationEffect.createOneShot(50, VibrationEffect.DEFAULT_AMPLITUDE))

            } else {

                @Suppress("DEPRECATION")

                vibrator.vibrate(50)

            }

        } catch (e: Exception) {

            Log.e(TAG, "Vibration error", e)

        }

    }



    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {

        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED

    }



    override fun onRequestPermissionsResult(

        requestCode: Int,

        permissions: Array<out String>,

        grantResults: IntArray

    ) {

        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (grantResults.all { it == PackageManager.PERMISSION_GRANTED }) {
                resetRecognizer()
            } else {
                Log.e(TAG, "Permissions not granted")
            }
        }
    }



    override fun onDestroy() {

        unregisterReceiver(usbReceiver)

        rsContext.close()

        super.onDestroy()

        detector.clear()

        cameraExecutor.shutdown()

        textToSpeech.stop()

        textToSpeech.shutdown()

        speechRecognizer.destroy()

        audioManager.abandonAudioFocusRequest(audioFocusRequest)

    }



    override fun onPause() {

        stopRealsensePipeline()

        stopListening()

        super.onPause()

    }



    override fun onResume() {

        super.onResume()

        if (isDetectionActive && rsContext.queryDevices().deviceCount > 0 && !isPipelineRunning) {
            startRealsensePipeline()
        }

        safeStartListening()

    }

    companion object {
        private const val TAG = "ObjectDetection"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private const val REQUEST_CODE_AUDIO_PERMISSION = 123
        private val REQUIRED_PERMISSIONS = arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.MODIFY_AUDIO_SETTINGS
        )
    }

}






/*package com.surendramaran.yolov8tflite

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.speech.RecognizerIntent
import android.speech.RecognitionListener
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.util.Log
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.surendramaran.yolov8tflite.Constants.LABELS_PATH
import com.surendramaran.yolov8tflite.Constants.MODEL_PATH
import com.surendramaran.yolov8tflite.databinding.ActivityMainBinding
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
// Add this to the top with other imports
import android.os.Vibrator
import android.os.VibrationEffect
import android.content.Context
import android.os.Handler
import android.os.Looper


enum class DetectionMode {
    START_DETECTION, EXPLAIN_SURROUNDING
}

class MainActivity : AppCompatActivity(), Detector.DetectorListener {
    private lateinit var binding: ActivityMainBinding
    private val isFrontCamera = false

    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private lateinit var detector: Detector
    private var isDetectionActive = false
    private lateinit var cameraExecutor: ExecutorService

    private lateinit var textToSpeech: TextToSpeech
    private var isTTSInitialized = false

    private lateinit var speechRecognizer: SpeechRecognizer
    private lateinit var speechRecognizerIntent: Intent
    private lateinit var vibrator: Vibrator
    private var currentMode: DetectionMode = DetectionMode.START_DETECTION
    private var isProcessingCommand = false


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        vibrator = getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        detector = Detector(baseContext, MODEL_PATH, LABELS_PATH, this)
        detector.setup()

        // Initialize TTS
        textToSpeech = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val result = textToSpeech.setLanguage(Locale.US)
                isTTSInitialized = result != TextToSpeech.LANG_MISSING_DATA && result != TextToSpeech.LANG_NOT_SUPPORTED
            } else {
                Log.e(TAG, "TTS Initialization failed")
            }
        }

        // Setup Speech Recognition
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this)
        speechRecognizerIntent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
        }

        speechRecognizer.setRecognitionListener(object : RecognitionListener {
            override fun onResults(results: Bundle?) {
                val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                matches?.let {
                    val spokenText = it[0].lowercase(Locale.ROOT)

                    if (!isProcessingCommand) {
                        isProcessingCommand = true // Mark as processing command

                        if ("start detection" in spokenText && !isDetectionActive) {
                            if (allPermissionsGranted()) {
                                if (isTTSInitialized) {
                                    textToSpeech.speak("Starting detection now", TextToSpeech.QUEUE_FLUSH, null, null)
                                }
                                startCamera()
                                isDetectionActive = true
                                currentMode = DetectionMode.START_DETECTION
                            }
                        } else if ("stop detection" in spokenText && isDetectionActive) {
                            if (isTTSInitialized) {
                                textToSpeech.speak("Stopping detection", TextToSpeech.QUEUE_FLUSH, null, null)
                            }
                            stopCamera()
                            isDetectionActive = false
                        } else if ("explain" in spokenText) {
                            if (isTTSInitialized) {
                                textToSpeech.speak("Explaining the surroundings", TextToSpeech.QUEUE_FLUSH, null, null)
                            }
                            currentMode = DetectionMode.EXPLAIN_SURROUNDING
                        }

                        // Reset the flag after a short delay (to allow speech to finish)
                        Handler(Looper.getMainLooper()).postDelayed({
                            isProcessingCommand = false
                            startListening()  // Restart listening after a delay
                        }, 1500)  // Delay of 1.5 seconds before accepting the next command
                    }
                }
            }


            override fun onError(error: Int) {
                startListening()  // Restart listening on error
            }

            override fun onReadyForSpeech(params: Bundle?) {}
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() {}
            override fun onPartialResults(partialResults: Bundle?) {}
            override fun onEvent(eventType: Int, params: Bundle?) {}
        })

        startListening() // Start listening for voice commands as soon as the app is launched

        if (!allPermissionsGranted()) {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    private fun startListening() {
        if (!isProcessingCommand) {
            speechRecognizer.startListening(speechRecognizerIntent)
        }
    }

    private fun stopListening() {
        speechRecognizer.stopListening()
    }


    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun stopCamera() {
        camera?.let {
            cameraProvider?.unbindAll()
            camera = null
        }
    }

    private fun bindCameraUseCases() {
        val cameraProvider = cameraProvider ?: throw IllegalStateException("Camera initialization failed.")

        val rotation = binding.viewFinder.display.rotation

        val cameraSelector = CameraSelector
            .Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .build()

        preview = Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(rotation)
            .build()

        imageAnalyzer = ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setTargetRotation(binding.viewFinder.display.rotation)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()

        imageAnalyzer?.setAnalyzer(cameraExecutor) { imageProxy ->
            val bitmapBuffer = Bitmap.createBitmap(imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888)
            imageProxy.use { bitmapBuffer.copyPixelsFromBuffer(imageProxy.planes[0].buffer) }
            imageProxy.close()

            val matrix = Matrix().apply {
                postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
                if (isFrontCamera) {
                    postScale(-1f, 1f, imageProxy.width.toFloat(), imageProxy.height.toFloat())
                }
            }

            val rotatedBitmap = Bitmap.createBitmap(
                bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height, matrix, true
            )

            detector.detect(rotatedBitmap)
        }

        cameraProvider.unbindAll()

        try {
            camera = cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)
            preview?.setSurfaceProvider(binding.viewFinder.surfaceProvider)
        } catch (exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) {
        if (it[Manifest.permission.CAMERA] == true && it[Manifest.permission.RECORD_AUDIO] == true) {
            startCamera()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        detector.clear()
        cameraExecutor.shutdown()

        textToSpeech.stop()
        textToSpeech.shutdown()

        speechRecognizer.destroy()
    }

    override fun onResume() {
        super.onResume()
        if (!allPermissionsGranted()) {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }
    }

    override fun onEmptyDetect() {
        binding.overlay.invalidate()
    }

    override fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long) {
        runOnUiThread {
            binding.inferenceTime.text = "${inferenceTime}ms"
            binding.overlay.apply {
                setResults(boundingBoxes)
                invalidate()
            }

            if (isTTSInitialized && boundingBoxes.isNotEmpty()) {
                val screenWidth = binding.viewFinder.width
                val screenHeight = binding.viewFinder.height

                // Filter bounding boxes with confidence > 50%
                val filteredBoundingBoxes = boundingBoxes.filter { box ->
                    box.cnf > 0.75 // Using 'cnf' as the confidence property
                }

                if (currentMode == DetectionMode.START_DETECTION) {
                    val closestObject = filteredBoundingBoxes.minByOrNull { box ->
                        val centerX = (box.x1 + box.x2) / 2
                        val centerY = (box.y1 + box.y2) / 2
                        val screenCenterX = screenWidth / 2
                        val screenCenterY = screenHeight / 2
                        // Calculate distance to center of screen
                        Math.abs(centerX - screenCenterX) + Math.abs(centerY - screenCenterY)
                    }

                    closestObject?.let { box ->
                        val label = box.clsName
                        val message = "I see a $label."
                        textToSpeech.speak(message, TextToSpeech.QUEUE_FLUSH, null, null)
                    }

                } else if (currentMode == DetectionMode.EXPLAIN_SURROUNDING) {
                    // Collect descriptions for each detected object
                    val descriptions = filteredBoundingBoxes.map { box ->
                        val label = box.clsName
                        val centerX = (box.x1 + box.x2) / 2
                        val direction = when {
                            centerX < screenWidth / 3 -> "on your left"
                            centerX > 2 * screenWidth / 3 -> "on your right"
                            else -> "ahead"
                        }
                        val area = (box.x2 - box.x1) * (box.y2 - box.y1)
                        val proximity = when {
                            area > screenWidth * screenHeight * 0.15 -> {
                                "very close"
                            }
                            area > screenWidth * screenHeight * 0.05 -> "close"
                            else -> "at a distance"
                        }

                        "$label $direction, $proximity"
                    }

                    // Join descriptions with commas
                    val message = descriptions.joinToString(", ")
                    textToSpeech.speak(message, TextToSpeech.QUEUE_FLUSH, null, null)
                }
            }
        }
    }


    companion object {
        private const val TAG = "Camera"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO
        )
    }
}
*/
/*
package com.surendramaran.yolov8tflite

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.speech.RecognizerIntent
import android.speech.RecognitionListener
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.util.Log
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.surendramaran.yolov8tflite.Constants.LABELS_PATH
import com.surendramaran.yolov8tflite.Constants.MODEL_PATH
import com.surendramaran.yolov8tflite.databinding.ActivityMainBinding
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), Detector.DetectorListener {
    private lateinit var binding: ActivityMainBinding
    private val isFrontCamera = false

    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private lateinit var detector: Detector

    private lateinit var cameraExecutor: ExecutorService

    private lateinit var textToSpeech: TextToSpeech
    private var isTTSInitialized = false

    private lateinit var speechRecognizer: SpeechRecognizer
    private lateinit var speechRecognizerIntent: Intent

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        detector = Detector(baseContext, MODEL_PATH, LABELS_PATH, this)
        detector.setup()

        // Initialize TTS
        textToSpeech = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val result = textToSpeech.setLanguage(Locale.US)
                isTTSInitialized = result != TextToSpeech.LANG_MISSING_DATA && result != TextToSpeech.LANG_NOT_SUPPORTED
            } else {
                Log.e(TAG, "TTS Initialization failed")
            }
        }

        // Setup Speech Recognition
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this)
        speechRecognizerIntent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
        }

        speechRecognizer.setRecognitionListener(object : RecognitionListener {
            override fun onResults(results: Bundle?) {
                val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                matches?.let {
                    val spokenText = it[0].lowercase(Locale.ROOT)
                    if ("start detection" in spokenText) {
                        if (isTTSInitialized) {
                            textToSpeech.speak("Starting detection now", TextToSpeech.QUEUE_FLUSH, null, null)
                        }
                        startCamera() // Starts the camera
                    } else if ("stop detection" in spokenText) {
                        if (isTTSInitialized) {
                            textToSpeech.speak("Stopping detection", TextToSpeech.QUEUE_FLUSH, null, null)
                        }
                        stopCamera() // Stop the camera
                    }
                }
                startListening() // Restart listening after command
            }

            override fun onError(error: Int) {
                startListening() // Restart listening on error
            }

            override fun onReadyForSpeech(params: Bundle?) {}
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() {}
            override fun onPartialResults(partialResults: Bundle?) {}
            override fun onEvent(eventType: Int, params: Bundle?) {}
        })

        startListening() // Start listening for voice commands as soon as the app is launched

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    private fun startListening() {
        speechRecognizer.startListening(speechRecognizerIntent)
    }

    private fun stopListening() {
        speechRecognizer.stopListening()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun stopCamera() {
        camera?.let {
            cameraProvider?.unbindAll()
            camera = null
        }
    }

    private fun bindCameraUseCases() {
        val cameraProvider = cameraProvider ?: throw IllegalStateException("Camera initialization failed.")

        val rotation = binding.viewFinder.display.rotation

        val cameraSelector = CameraSelector
            .Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .build()

        preview = Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(rotation)
            .build()

        imageAnalyzer = ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setTargetRotation(binding.viewFinder.display.rotation)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()

        imageAnalyzer?.setAnalyzer(cameraExecutor) { imageProxy ->
            val bitmapBuffer = Bitmap.createBitmap(imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888)
            imageProxy.use { bitmapBuffer.copyPixelsFromBuffer(imageProxy.planes[0].buffer) }
            imageProxy.close()

            val matrix = Matrix().apply {
                postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
                if (isFrontCamera) {
                    postScale(-1f, 1f, imageProxy.width.toFloat(), imageProxy.height.toFloat())
                }
            }

            val rotatedBitmap = Bitmap.createBitmap(
                bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height, matrix, true
            )

            detector.detect(rotatedBitmap)
        }

        cameraProvider.unbindAll()

        try {
            camera = cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)
            preview?.setSurfaceProvider(binding.viewFinder.surfaceProvider)
        } catch (exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) {
        if (it[Manifest.permission.CAMERA] == true && it[Manifest.permission.RECORD_AUDIO] == true) {
            startCamera()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        detector.clear()
        cameraExecutor.shutdown()

        textToSpeech.stop()
        textToSpeech.shutdown()

        speechRecognizer.destroy()
    }

    override fun onResume() {
        super.onResume()
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }
    }

    override fun onEmptyDetect() {
        binding.overlay.invalidate()
    }

    override fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long) {
        runOnUiThread {
            binding.inferenceTime.text = "${inferenceTime}ms"
            binding.overlay.apply {
                setResults(boundingBoxes)
                invalidate()
            }

            if (isTTSInitialized && boundingBoxes.isNotEmpty()) {
                val detectedLabels = boundingBoxes.joinToString(", ") { it.clsName }
                textToSpeech.speak("I see: $detectedLabels", TextToSpeech.QUEUE_FLUSH, null, null)
            }
        }
    }

    companion object {
        private const val TAG = "Camera"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO
        )
    }
}


//    private fun processFrame(frames: FrameSet) {
//
//        if(hasSavedSingleFrame) return
//
//        frames.first(StreamType.COLOR)?.use { colorFrame ->
//            if (colorFrame.`is`(Extension.VIDEO_FRAME)) {
//                val videoFrame = colorFrame.`as`<VideoFrame>(Extension.VIDEO_FRAME)
//                val bitmap = videoFrame.toBitmap()
//
//                saveBitmapAsPng(bitmap, "SingleFrame")
//
//                val resized = Bitmap.createScaledBitmap(bitmap, 640, 640, true)
//
//                saveBitmapAsPng(resized, "ResizedSingleFrame")
//
//                detector.detect(resized)
//
//                hasSavedSingleFrame = true
//
//
//            }
//        }
//    }
//
//    private fun saveBitmapAsPng(bitmap: Bitmap, fileName: String) {
//        try {
//            val dir = File(getExternalFilesDir(null), "SingleFrame")
//            if (!dir.exists()) {
//                dir.mkdirs()
//            }
//
//            val file = File(dir, "$fileName.png")
//            FileOutputStream(file).use { out ->
//                val success = bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
//                out.flush()
//                if (success) {
//                    Log.d("YOLO", "Saved frame to ${file.absolutePath}")
//                } else {
//                    Log.e("YOLO", "Bitmap.compress() failed")
//                }
//            }
//        } catch (e: Exception) {
//            Log.e("YOLO", "Failed to output bitmap", e)
//        }
//    }
*/

