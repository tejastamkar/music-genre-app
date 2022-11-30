import 'dart:io';

import 'package:file_picker/file_picker.dart';
import 'package:musicgenre/data/filedata.dart';

Future getFile() async {
  FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.custom, allowMultiple: false, allowedExtensions: ['wav']);

  if (result != null) {
    File file = File("${result.files.single.path}");
    fileName = file;
    return file;
  } else {
    return "";
  }
}
