import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_app/main.dart';
import 'package:camera/camera.dart';


void main() {
  testWidgets('Live and Upload buttons are present', (WidgetTester tester) async {
    final cameras = <CameraDescription>[]; // Mock or provide actual if needed
    await tester.pumpWidget(SignApp(cameras: cameras));

    expect(find.text('Live Translate'), findsOneWidget);
    expect(find.text('Upload Video'), findsOneWidget);
  });
}
