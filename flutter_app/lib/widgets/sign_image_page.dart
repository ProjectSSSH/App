import 'package:flutter/material.dart';
import '../utils/constants.dart';

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
