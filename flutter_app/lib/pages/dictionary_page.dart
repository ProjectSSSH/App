import 'package:flutter/material.dart';
import '../utils/constants.dart';
import '../widgets/sign_image_page.dart';
import 'package:url_launcher/url_launcher.dart';

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
