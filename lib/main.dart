import 'package:flutter/material.dart';
import 'package:musicgenre/homescreen.dart';

Future<void> main() async {
  runApp(MaterialApp(
    title: 'MyApp',
    theme: ThemeData(
        scaffoldBackgroundColor: Colors.white, backgroundColor: Colors.white),
    home: const Main(),
  ));
}

class Main extends StatelessWidget {
  const Main({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return const HomeScreen();
  }
}
