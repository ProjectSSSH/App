import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:video_player/video_player.dart';



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
    final uri = Uri.parse("http://192.168.201.170:5000/predict_video");
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