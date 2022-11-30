import 'package:flutter/material.dart';
import 'package:musicgenre/api/genre.dart';
import 'package:musicgenre/provider/files.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  String genre = '';
  bool fileSelected = false;

  callFunc() async {
    var temp = await getGenre();
    setState(() {
      genre = temp;
    });
  }

  @override
  Widget build(BuildContext context) {
    double width = MediaQuery.of(context).size.width;

    return Scaffold(
      appBar: AppBar(
          elevation: 0,
          centerTitle: true,
          foregroundColor: Colors.black,
          backgroundColor: Colors.white,
          title: const Text(
            "Music Genre",
            style: TextStyle(fontSize: 20, fontWeight: FontWeight.w700),
          )),
      body: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: Center(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                const Text(
                  'Pick Wav Audio File:',
                  style: TextStyle(fontSize: 20, fontWeight: FontWeight.w700),
                ),
                InkWell(
                  onTap: () => getFile().whenComplete(() => setState(
                        () => fileSelected = true,
                      )),
                  child: Image.asset(
                    "assets/file.jpg",
                    width: width - 100,
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.symmetric(vertical: 30),
                  child: ElevatedButton(
                    style: ElevatedButton.styleFrom(
                        foregroundColor: Colors.white,
                        backgroundColor: Colors.yellow.shade600,
                        fixedSize: Size(width, 50)),
                    onPressed: () => fileSelected ? callFunc() : null,
                    child: const Text(
                      'Get Music Genre',
                      style:
                          TextStyle(fontSize: 20, fontWeight: FontWeight.w700),
                    ),
                  ),
                ),
                genre == ""
                    ? const SizedBox.shrink()
                    : Text(
                        'The Genre Of The Given Music is: $genre',
                        style: const TextStyle(
                            fontSize: 16, fontWeight: FontWeight.w700),
                      ),
              ],
            ),
          )),
    );
  }
}
