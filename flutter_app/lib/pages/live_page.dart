import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:collection';
import 'package:socket_io_client/socket_io_client.dart' as IO;

class LivePage extends StatefulWidget {
  final CameraDescription camera;  // add this

  LivePage({required this.camera});  // constructor
  @override
  _LivePageState createState() => _LivePageState();
}

class _LivePageState extends State<LivePage> {
  late CameraController _controller;
  String _label = "";
  bool _streaming = false;
  int _cameraIndex = 0; // back=0, front=1
  late List<CameraDescription> _availableCameras;
  final Queue<String> _frameQueue = Queue<String>();
  bool _isSending = false;
  double _cameraRatio = 0.75;
  bool _enlarged = false;
  late IO.Socket _socket;
  int _frameCount = 0;
late DateTime _lastFpsTime;
double _fps = 0.0;


  @override
void initState() {
  super.initState();
  _initializeCameras();
  _connectSocket();
}


  Future<void> _initializeCameras() async {
    _availableCameras = await availableCameras();
    await _initializeController(_availableCameras[_cameraIndex]);
  }

  Future<void> _initializeController(CameraDescription cameraDescription) async {
    _controller = CameraController(cameraDescription, ResolutionPreset.medium);
    await _controller.initialize();
    if (mounted) setState(() {});
  }

  void _toggleCamera() async {
    _stopStream(); // Stop current streaming first
    _cameraIndex = (_cameraIndex + 1) % _availableCameras.length;
    await _initializeController(_availableCameras[_cameraIndex]);
  }

  void _startStopStream() {
    if (_streaming) {
      _stopStream();
    } else {
      _startStream();
    }
  }

  void _startStream() {
    if (_streaming) return;
    setState(() {
      _streaming = true;
      _frameCount = 0;
  _lastFpsTime = DateTime.now();
    });
    _controller.startImageStream((CameraImage image) {
      if (!_streaming) return;
      _frameCount++;
      final now = DateTime.now();
      final diff = now.difference(_lastFpsTime).inMilliseconds;
      if (diff >= 1000) {
        setState(() {
          _fps = _frameCount * 1000 / diff;
        });
        // _frameCount = 0;
        _lastFpsTime = now;
        print("FPS: $_fps");
      }
      _captureAndEnqueueFrame();
    });
  }

void _stopStream() async {
  if (!_streaming) return;

  await _controller.stopImageStream();

  setState(() {
    _streaming = false;
  });

  // Don't wait for _trySendNextFrame to recurse — loop until queue is empty
}
Future<void> _flushFrameQueue() async {
  while (_frameQueue.isNotEmpty) {
    if (!_isSending) {
      final imgPath = _frameQueue.removeFirst();
      await _sendFrameOverSocket(imgPath);
    } else {
      // Wait briefly before trying again
      await Future.delayed(Duration(milliseconds: 100));
    }
  }
}

  bool _isCapturing = false;
void _connectSocket() {
    _socket = IO.io('http://192.168.201.170:5000', <String, dynamic>{
      'transports': ['websocket'],
      'autoConnect': false,
    });

    _socket.connect();

    _socket.on('prediction', (data) {
      final prediction = data['prediction'] ?? "";
      setState(() {
        if (prediction!='No Gesture Detected'){
        _label += ' '+prediction;
        }
      });
      print('Received prediction: ${data['prediction']}');
    });

    _socket.onDisconnect((_) => print('Disconnected'));
  }

// Future<void> _sendFrameOverSocket(String imgPath) async {
//     if (_isSending) return;
//     _isSending = true;

//     try {
//       final bytes = await File(imgPath).readAsBytes();
//       _socket.emit('frame', bytes);
//     } catch (e) {
//       print("Error sending frame over WebSocket: $e");
//     } finally {
//       _isSending = false;
//       try {
//         await File(imgPath).delete();
//       } catch (_) {}
//     }
//   }
Future<void> _sendFrameOverSocket(String imgPath) async {
  try {
    debugPrint("sending: ${_frameQueue.length}");

    final bytes = await File(imgPath).readAsBytes();
    _socket.emit('frame', bytes);
  } catch (e) {
    print("Error sending frame over WebSocket: $e");
  } finally {
    try {
      await File(imgPath).delete();
    } catch (_) {}
  }
}

// Future<void> _captureAndEnqueueFrame() async {
//   if (!_controller.value.isInitialized || _isCapturing || !_streaming) return;

//   _isCapturing = true;
//   try {
//     final tempDir = await getTemporaryDirectory();
//     final imgPath = '${tempDir.path}/${DateTime.now().millisecondsSinceEpoch}.jpg';

//     final XFile file = await _controller.takePicture();
//     await file.saveTo(imgPath);

//     _frameQueue.add(imgPath);
//     _trySendNextFrame();
//   } catch (e) {
//     print("Error capturing or sending frame: $e");
//   } finally {
//     _isCapturing = false;
//   }
// }
Future<void> _captureAndEnqueueFrame() async {
  if (!_controller.value.isInitialized || _isCapturing || !_streaming) return;
debugPrint("captureenqueue: ${_frameQueue.length}");

  _isCapturing = true;
  try {
    final tempDir = await getTemporaryDirectory();
    final imgPath = '${tempDir.path}/${DateTime.now().millisecondsSinceEpoch}.jpg';

    final XFile file = await _controller.takePicture();
    await file.saveTo(imgPath);

    _frameQueue.add(imgPath);
    _trySendNextFrame();  // Always try to start sending
  } catch (e) {
    print("Error capturing or sending frame: $e");
  } finally {
    _isCapturing = false;
  }
}


// void _trySendNextFrame() async {
//   if (_isSending || _frameQueue.isEmpty) return;

//   final imgPath = _frameQueue.removeFirst();
//   await _sendFrameOverSocket(imgPath);

//   // Don't check _streaming here — keep flushing
//   _trySendNextFrame();
// }
// void _trySendNextFrame() async {
//   if (_isSending) return;
//   _isSending = true;

//   while (_frameQueue.isNotEmpty) {
//     final imgPath = _frameQueue.removeFirst();
//     try {
//       await _sendFrameOverSocket(imgPath);
//       await Future.delayed(Duration(milliseconds: 10));
//     } catch (e) {
//       print("Error sending frame: $e");
//     }
//   }

//   _isSending = false;
// }
void _trySendNextFrame() async {
  debugPrint("Queue length before sending: ${_frameQueue.length}");

  if (_isSending) return; // Already sending, no need to start again
  _isSending = true;

  Future.doWhile(() async {
    if (_frameQueue.isEmpty) {
      _isSending = false;
      return false; // Stop the loop
    }

    final imgPath = _frameQueue.removeFirst();
    try {
      await _sendFrameOverSocket(imgPath);
    } catch (e) {
      print("Error sending frame: $e");
    }

    await Future.delayed(Duration(milliseconds: 10));
    return true; // Continue while loop
  });
}



  @override
void dispose() {
  if (_streaming) {
    _stopStream();
  }
  _socket.dispose();
  _controller.dispose();
  super.dispose();
}

  @override
  Widget build(BuildContext context) {
    if (!_controller.value.isInitialized) {
      return Scaffold(body: Center(child: CircularProgressIndicator()));
    }
 
    return Scaffold(
      appBar: AppBar(
        title: Text("Live Translation"),
        actions: [
          // Start/Stop Button
          IconButton(
            icon: Icon(_streaming ? Icons.stop : Icons.play_arrow),
            onPressed: _startStopStream,
          ),
          // Camera Flip Button
          IconButton(
            icon: Icon(Icons.switch_camera),
            onPressed: _toggleCamera,
          ),
        ],
      ),
      body: Column(
        children: [
          if (!_enlarged)
            SizedBox(
                height: MediaQuery.of(context).size.height * _cameraRatio,
                child: Stack(
    children: [
      Transform(
        alignment: Alignment.center,
        transform: _availableCameras[_cameraIndex].lensDirection == CameraLensDirection.front
            ? Matrix4.rotationY(math.pi)
            : Matrix4.identity(),
        child: CameraPreview(_controller),
      ),
      Positioned(
        top: 20,
        left: 20,
        child: Container(
          padding: EdgeInsets.all(6),
          color: Colors.black54,
          child: Text(
            // 'FPS: ${_fps.toStringAsFixed(2)}',
            'frames: ${_frameCount.toStringAsFixed(2)}',
            style: TextStyle(color: Colors.white, fontSize: 16),
          ),
        ),
      ),
    ],
  ),
),

          if (!_enlarged)
            GestureDetector(
              onVerticalDragUpdate: (details) {
                setState(() {
                  _cameraRatio += details.primaryDelta! / MediaQuery.of(context).size.height;
                  _cameraRatio = _cameraRatio.clamp(0.3, 0.9);
                });
              },
              child: Container(
                height: 15,
                color: Colors.grey.shade600,
                child: Center(child: Container(height: 10, width: 100, color: Colors.white)),
              ),
            ),

          // Fullscreen Button and Sign Language Translation at the bottom
          Expanded(
            child: Container(
              width: MediaQuery.of(context).size.width * 0.9,  // 90% of the screen width
              color: const Color.fromARGB(255, 253, 251, 251),
              padding: EdgeInsets.all(15),
              child: Column(
                children: [
                  Align(
                    alignment: Alignment.topRight,
                    child: IconButton(
                      icon: Icon(_enlarged ? Icons.fullscreen_exit : Icons.fullscreen),
                      onPressed: () {
                        setState(() {
                          _enlarged = !_enlarged;  // Toggle full-screen mode
                        });
                      },
                    ),
                  ),
                  Expanded(
                    child: SingleChildScrollView(
                      child: SelectableText(
                        _label.isEmpty ? "No sign detected" : _label,  // Display _label for translation
                        style: TextStyle(color: const Color.fromARGB(255, 7, 0, 0), fontSize: 20),
                       ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
