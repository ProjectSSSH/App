import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'pages/home_page.dart';


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