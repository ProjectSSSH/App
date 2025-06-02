import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'dictionary_page.dart';
import 'live_page.dart';
import 'video_prediction_page.dart';
import 'package:file_picker/file_picker.dart';
import 'dart:io';

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
