import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;
import 'package:video_player/video_player.dart';
import 'package:file_picker/file_picker.dart';
import 'package:image/image.dart' as img;
import 'dart:collection';
import 'package:url_launcher/url_launcher.dart';


void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  runApp(SignApp(cameras: cameras));
}

class SignApp extends StatelessWidget {
  final List<CameraDescription> cameras;
  SignApp({required this.cameras});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: HomePage(cameras: cameras),
      debugShowCheckedModeBanner: false,
    );
  }
}

class HomePage extends StatelessWidget {
  final List<CameraDescription> cameras;
  HomePage({required this.cameras});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        centerTitle: true,
        toolbarHeight: 170,
        title: Padding(
          padding: const EdgeInsets.only(top: 60.0), // Adjust the value as needed
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text('Indian Sign Language',style: TextStyle(fontSize: 25)),
              SizedBox(height: 15),
              Text('Translation App',style: TextStyle(fontSize: 25)),
            ],
          ),
        ),
      ),

   body: Center(
      child: Padding(
        padding: const EdgeInsets.only(bottom: 120.0), // Move buttons higher
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton(
              child: Text("Dictionary"),
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => DictionaryPage()),
                );
              },
            ),
            SizedBox(height: 20),
            ElevatedButton(
              child: Text("Live Translate"),
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => LivePage(camera: cameras[0])),
                );
              },
            ),
            SizedBox(height: 20),
            ElevatedButton(
              child: Text("Upload Video"),
              onPressed: () async {
                FilePickerResult? result = await FilePicker.platform.pickFiles(
                  type: FileType.video,
                  allowMultiple: false,
                );

                if (result != null && result.files.single.path != null) {
                  File videoFile = File(result.files.single.path!);
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (_) => VideoPredictionPage(videoFile: videoFile),
                    ),
                  );
                } else {
                  // User canceled or no video picked
                }
              },
            ),
            SizedBox(height: 20),
          ],
        ),
      ),
   ),
    );
  }
}
final Map<String, String> alphabetDictionary = {
  'A': 'assets/A.png',
  'B': 'assets/B.png',
  'C': 'assets/C.png',
  'D': 'assets/D.png',
  'E': 'assets/E.png',
  'F': 'assets/F.png',
  'G': 'assets/G.png',
  'H': 'assets/H.png',
  'I': 'assets/I.png',
  'J': 'assets/J.png',
  'K': 'assets/K.png',
  'L': 'assets/L.png',
  'M': 'assets/M.png',
  'N': 'assets/N.png',
  'O': 'assets/O.png',
  'P': 'assets/P.png',
  'Q': 'assets/Q.png',
  'R': 'assets/R.png',
  'S': 'assets/S.png',
  'T': 'assets/T.png',
  'U': 'assets/U.png',
  'V': 'assets/V.png',
  'W': 'assets/W.png',
  'X': 'assets/X.png',
  'Y': 'assets/Y.png',
  'Z': 'assets/Z.png',
  '1': 'assets/1.png',
  '2': 'assets/2.png',
  '3': 'assets/3.png',
  '4': 'assets/4.png',
  '5': 'assets/5.png',
  '6': 'assets/6.png',
  '7': 'assets/7.png',
  '8': 'assets/8.png',
  '9': 'assets/9.png'
};

class DictionaryPage extends StatelessWidget {
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Alphabet Dictionary"),
      ),
      body:Padding(
        padding: const EdgeInsets.all(8.0),
        child: Column(
          children: [
            // Link above the GridView
            TextButton(
              onPressed: () {
                // Navigate to the dictionary URL or open a webpage
                launchURL();
              },
              child: Text(
                'Go to the Word Dictionary : https://indiansignlanguage.org/',
                style: TextStyle(fontSize: 16, color: Colors.blue),
              ),
            ),
            SizedBox(height: 20), // Space between the link and the GridView
            Expanded(
              child: GridView.builder(
            gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
              crossAxisCount: 4, // 4 columns for the alphabet
              childAspectRatio: 1, // Make grid items square
            ),
            itemCount: alphabetDictionary.keys.length,
            itemBuilder: (context, index) {
              String letter = alphabetDictionary.keys.elementAt(index);
              return GestureDetector(
                onTap: () {
                  // Show the image of the selected letter
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => SignImagePage(letter: letter),
                    ),
                  );
                },
                child: Card(
                  margin: EdgeInsets.all(10),
                  child: Center(
                    child: Text(
                      letter,
                      style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                    ),
                  ),
                ),
              );
            },
          ),
            ),
          ],
        ),
      ),
    );
  }
}
void launchURL() async {
    final Uri _url = Uri.parse('https://indiansignlanguage.org/');
    if (await canLaunchUrl(_url)) {
      await launchUrl(_url);
    } else {
      throw 'Could not launch $_url';
    }
  }

class SignImagePage extends StatelessWidget {
  final String letter;
  SignImagePage({required this.letter});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('$letter Sign'),
      ),
      body: Center(
        child: Image.asset(alphabetDictionary[letter]!),
      ),
    );
  }
}


class LivePage extends StatefulWidget {
  final CameraDescription camera;
  LivePage({required this.camera});

  @override
  _LivePageState createState() => _LivePageState();
}

class _LivePageState extends State<LivePage> {
  late CameraController _controller;
  String _label = "";
  bool _streaming = false;
  Timer? _timer;
  double _cameraRatio = 0.75;
  bool _enlarged = false;
  int _cameraIndex = 0; // 0 = back, 1 = front
  late List<CameraDescription> _availableCameras;
  final Queue<String> _frameQueue = Queue<String>();
  bool _isSending = false;
  @override
  void initState() {
    super.initState();
    _initializeCameras();
  }
  Future<void> _initializeCameras() async {
    _availableCameras = await availableCameras();
    _initializeController(_availableCameras[_cameraIndex]);
  }
  void _initializeController(CameraDescription cameraDescription) async {
    _controller = CameraController(cameraDescription, ResolutionPreset.medium);
    await _controller.initialize();
    if (mounted) setState(() {});
  }
  void _toggleCamera() {
    _cameraIndex = (_cameraIndex + 1) % _availableCameras.length;
    _stopStream(); // Stop stream when switching cameras
    _initializeController(_availableCameras[_cameraIndex]);
  }
  void _startStopStream() async {
    if (_streaming) {
      _stopStream();
    } else {
      setState(() {}); // Immediate feedback
      _startStream();
    }
  }
  void _startStream() async {
    if (_streaming) return;
    setState(() {
      _streaming = true;
    });
    _timer = Timer.periodic(Duration(milliseconds: 40), (_) {
      _captureAndEnqueueFrame();
    });
    await _controller.startImageStream((CameraImage image) {
      if (!_streaming) return; // If stopped, ignore new frames
      // We add the image frame as a path string
      _captureAndEnqueueFrame();
    });
  }
  void _stopStream() {
    setState(() {
      _streaming = false;
    });
  }
  Future<void> _captureAndEnqueueFrame() async {
    if (!_controller.value.isInitialized || !_streaming) return;

    try {
      // Capture a frame as a JPEG file and save it
      final tempDir = await getTemporaryDirectory();
      final imgPath = path.join(tempDir.path, '${DateTime.now().millisecondsSinceEpoch}.jpg');
      
      // Capture the image and save it as JPEG
      final XFile file = await _controller.takePicture();
      await file.saveTo(imgPath);

      // Add the file path to the queue
      _frameQueue.add(imgPath);
      _trySendNextFrame(); // Immediately try to process the frame
    } catch (e) {
      print("Error capturing frame: $e");
    }
  }

  // Try to send the next frame from the queue
  void _trySendNextFrame() async {
    if (_isSending) return; // Prevent sending multiple frames simultaneously
    if (_frameQueue.isEmpty) return; // No frames to send

    _isSending = true;
    final imgPath = _frameQueue.removeFirst();

    _sendFrame(imgPath).whenComplete(() {
      _isSending = false;
      _trySendNextFrame(); // Continue processing if there are more frames
    });
  }

  // Send the frame to the backend (Flask server)
  Future<void> _sendFrame(String imgPath) async {
    try {
      final request = http.MultipartRequest(
        'POST',
        Uri.parse("http://192.168.103.170:5000/predict_frame"),
      );
      request.files.add(await http.MultipartFile.fromPath('file', imgPath));

      final response = await request.send();
      final json = jsonDecode(await response.stream.bytesToString());

      if (mounted) {
        setState(() {
          _label += (json['label']?.trim().isNotEmpty ?? false) ? json['label'] : " ";
        });
      }
    } catch (e) {
      print("Error sending frame: $e");
    } finally {
      try {
        File(imgPath).delete(); // Clean up the temporary file
      } catch (e) {
        print("Error deleting frame: $e");
      }
    }
  }

  @override
  void dispose() {
    if (_streaming) {
      _stopStream(); // Ensure stopping the stream if it's active
    }
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_controller.value.isInitialized && !_enlarged) {
      return Scaffold(body: Center(child: CircularProgressIndicator()));
    }


// class _LivePageState extends State<LivePage> {
//   late CameraController _controller;
//   String _label = "";
//   bool _streaming = false;
//   double _cameraRatio = 0.75;
//   bool _enlarged = false;
//   int _cameraIndex = 0; // 0 = back, 1 = front
//   late List<CameraDescription> _availableCameras;

//   // New: Queue and sending flag
//   Queue<CameraImage> _frameQueue = Queue<CameraImage>();
//   bool _sendingInProgress = false;
//   bool _isSending = false;

//   @override
//   void initState() {
//     super.initState();
//     _initializeCameras();
//   }

//   Future<void> _initializeCameras() async {
//     _availableCameras = await availableCameras();
//     _initializeController(_availableCameras[_cameraIndex]);
//   }

//   void _initializeController(CameraDescription cameraDescription) async {
//     _controller = CameraController(
//       cameraDescription,
//       ResolutionPreset.medium,
//       imageFormatGroup: ImageFormatGroup.yuv420,
//     );
//     await _controller.initialize();
//     if (mounted) setState(() {});
//   }

//   void _toggleCamera() {
//     _cameraIndex = (_cameraIndex + 1) % _availableCameras.length;
//     _stopStream();
//     _initializeController(_availableCameras[_cameraIndex]);
//   }

//   void _startStopStream() async {
//     if (_streaming) {
//       _stopStream();
//     } else {
//       setState(() {}); // Immediate feedback
//       _startStream();
//     }
//   }

//   void _startStream() async {
//     if (_streaming) return;

//     setState(() {
//       _streaming = true;
//     });

//     await _controller.startImageStream((CameraImage image) {
//       if (!_streaming) return; // If stopped, ignore new frames
//       _frameQueue.add(image);
//       _trySendNextFrame();
//     });
//   }

//   void _trySendNextFrame() async{
//     if (_sendingInProgress) return;
//     if (_frameQueue.isEmpty) return;

//     _sendingInProgress = true;
//     final frame = _frameQueue.removeFirst();

//     _sendFrame(frame).whenComplete(() {
//       _sendingInProgress = false;
//       _trySendNextFrame();
//     });
//   }

//   void _stopStream() async {
//   setState(() {
//     _streaming = false; // Stop accepting new frames immediately
//   });

//   // Wait until the queue is fully processed
//   while (_sendingInProgress || _frameQueue.isNotEmpty) {
//     await Future.delayed(Duration(milliseconds: 50));
//   }

//   // Only now stop the image stream safely
//   await _controller.stopImageStream();
// }


//   Future<void> _sendFrame(CameraImage image) async {
//   try {
//     // Extract the raw YUV420 data from the camera image
//     List<int> rawYUVData = _getYUV420Data(image);
    
//     // Send the raw YUV420 data to the backend (Flask server)
//     final request = http.MultipartRequest(
//   'POST',
//   Uri.parse("http://192.168.103.170:5000/predict_frame"),
// );

// request.files.add(http.MultipartFile.fromBytes(
//   'file', rawYUVData, filename: 'frame.yuv',
// ));


//     final response = await request.send();
//     final json = jsonDecode(await response.stream.bytesToString());

//     if (!_streaming && _frameQueue.isEmpty) return; // Stop updating label if fully stopped

//     setState(() {
//       _label += (json['label']?.trim().isNotEmpty ?? false) ? json['label'] : " ";
//     });
//   } catch (e) {
//     print("Error sending frame: $e");
//   }
// }

// List<int> _getYUV420Data(CameraImage image) {
//   final yPlane = image.planes[0].bytes;
//   final uPlane = image.planes[1].bytes;
//   final vPlane = image.planes[2].bytes;
//   // Some Android devices store U and V swapped (i.e., NV21 instead of I420)
//   // Ensure format matches what Flask expects (I420: Y + U + V)
//   // Check if U and V planes have correct stride & size (should be width/2 * height/2)
//   final int uvLength = uPlane.length;
//   // Combine in I420 format: Y + U + V
//   return [...yPlane, ...uPlane, ...vPlane];
// }


//   @override
// void dispose() {
//   if (_streaming) {
//     _stopStream(); 
//   }
//   _controller.dispose();
//   super.dispose();
// }


//   @override
//   Widget build(BuildContext context) {
//     if (!_controller.value.isInitialized && !_enlarged) {
//       return Scaffold(body: Center(child: CircularProgressIndicator()));
//     }
 
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
                child: Transform(
                  alignment: Alignment.center,
                  transform: _availableCameras[_cameraIndex].lensDirection == CameraLensDirection.front
                      ? Matrix4.rotationY(math.pi)
                      : Matrix4.identity(),
                  child: CameraPreview(_controller),
                ),
              ),
          if (!_enlarged)
            GestureDetector(
              onVerticalDragUpdate: (details) {
                setState(() {
                  _cameraRatio += details.primaryDelta! / MediaQuery.of(context).size.height;
                  _cameraRatio = _cameraRatio.clamp(0.5, 0.5);
                });
              },
              child: Container(
                height: 6,
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


class VideoPredictionPage extends StatefulWidget {
  final File videoFile;

  VideoPredictionPage({required this.videoFile});

  @override
  _VideoPredictionPageState createState() => _VideoPredictionPageState();
}

class _VideoPredictionPageState extends State<VideoPredictionPage> {
  late VideoPlayerController _videoController;
  bool _enlarged = false;
  double _videoRatio = 0.75;
  String _prediction = "";
  bool _isPlaying = false;

  @override
  void initState() {
    super.initState();
    _videoController = VideoPlayerController.file(widget.videoFile)
      ..initialize().then((_) {
        setState(() {});
        _videoController.play();
        _isPlaying = true;
      });

    _uploadAndGetPrediction();
  }
  void _togglePlayPause() {
    setState(() {
      if (_isPlaying) {
        _videoController.pause();
        _isPlaying = false;
      } else {
        _videoController.play();
        _isPlaying = true;
      }
    });
  }
  void _seekForward() {
    final currentPosition = _videoController.value.position;
    final maxPosition = _videoController.value.duration;
    final newPosition = currentPosition + Duration(seconds: 10);

    if (newPosition < maxPosition) {
      _videoController.seekTo(newPosition);
    } else {
      _videoController.seekTo(maxPosition); // Ensure it doesn't go beyond the video duration
    }
  }
  void _seekBackward() {
    final currentPosition = _videoController.value.position;
    final newPosition = currentPosition - Duration(seconds: 10);

    if (newPosition > Duration.zero) {
      _videoController.seekTo(newPosition);
    } else {
      _videoController.seekTo(Duration.zero); // Ensure it doesn't go below zero
    }
  }

  Future<void> _uploadAndGetPrediction() async {
    final uri = Uri.parse("http://192.168.103.170:5000/predict_video");
    final request = http.MultipartRequest('POST', uri);
    request.files.add(await http.MultipartFile.fromPath('video', widget.videoFile.path));
    final response = await request.send();
    final responseBody = await response.stream.bytesToString();
    final result = jsonDecode(responseBody)['prediction'];

    setState(() {
      _prediction = result;
    });
  }

  @override
  void dispose() {
    _videoController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Video Translation"),
      ),
      body: Column(
        children: [
          if (!_enlarged)
            SizedBox(
              height: MediaQuery.of(context).size.height * _videoRatio,
              child: _videoController.value.isInitialized
                  ? AspectRatio(
                      aspectRatio: _videoController.value.aspectRatio,
                      child: VideoPlayer(_videoController),
                    )
                  : Center(child: CircularProgressIndicator()),
            ),

          if (!_enlarged)
            GestureDetector(
              onVerticalDragUpdate: (details) {
                setState(() {
                  _videoRatio += details.delta.dy / MediaQuery.of(context).size.height;
                  _videoRatio = _videoRatio.clamp(0.3, 0.9);
                });
              },
              child: Container(
                height: 6,
                color: Colors.grey.shade600,
                child: Center(child: Container(height: 2, width: 100, color: Colors.white)),
              ),
            ),

          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              IconButton(
                icon: Icon(Icons.replay_10),
                onPressed: _seekBackward,
              ),
              IconButton(
                icon: Icon(_isPlaying ? Icons.pause : Icons.play_arrow), // Change icon based on state
                onPressed: _togglePlayPause,
              ),
              IconButton(
                icon: Icon(Icons.forward_10),
                onPressed: _seekForward,
              ),
              IconButton(
                icon: Icon(_enlarged ? Icons.fullscreen_exit : Icons.fullscreen),
                onPressed: () {
                  setState(() {
                    _enlarged = !_enlarged;
                  });
                },
              ),
            ],
          ),
 Expanded(
            child: Container(
              color: Colors.white,
              padding: EdgeInsets.all(25),
              child: Stack(
                children: [
                  SingleChildScrollView(
                    child: Container(
                      width: MediaQuery.of(context).size.width * 0.9,  // 90% of screen width
                      child: SelectableText(
                        _prediction.isEmpty ? "Processing..." : _prediction,
                        style: TextStyle(fontSize: 20),
                      ),
                    ),
                  ),
                  Positioned(
                    bottom: 10,
                    right: 10,
                    child: IconButton(
                      icon: Icon(_enlarged ? Icons.fullscreen_exit : Icons.fullscreen),
                      onPressed: () {
                        setState(() {
                          _enlarged = !_enlarged;
                        });
                      },
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