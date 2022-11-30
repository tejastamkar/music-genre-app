import 'package:http/http.dart' as http;
import 'package:musicgenre/data/filedata.dart';

Future getGenre() async {
  String url = "http://192.168.29.93:5000/genre";

  var request = http.MultipartRequest(
    'POST',
    Uri.parse(url),
  );

  request.files.add(await http.MultipartFile.fromPath('files', fileName.path));
  // var res = await request.send();
  http.Response response = await http.Response.fromStream(await request.send());
  print(response.body);
  return response.body;
}
