import java.io.*;

// This function returns all the files in a directory as an array of Strings  
String[] listFileNames(String dir) {
  File file = new File(dir);
  if (file.isDirectory()) {
    String names[] = file.list();
    return names;
  } else {
    // If it's not a directory
    return null;
  }
}

// function to create the string array to be called by the system
// be sure that 'generatorScript', 'modelsDirectory',
// 'modelIndex', 'nbWords' and 'primeText' have been 
// set up correctly before calling this function
public void execScriptCommand() {
  StringList scriptArgs = new StringList();
  scriptArgs.append("python");
  scriptArgs.append(generatorScript);
  scriptArgs.append("--save_dir");
  scriptArgs.append(modelsDirectory);
  scriptArgs.append("--model");
  scriptArgs.append(modelNames[modelIndex]);
  scriptArgs.append("-n");
  scriptArgs.append(str(nbWords));
  scriptArgs.append("--prime");
  scriptArgs.append("\"" + primeText + "\"");
  scriptArgs.append("--balance");
  scriptArgs.append("1");
  scriptArgs.append("--osc_port");
  scriptArgs.append(str(listeningPort));
  execSystemCommand(scriptArgs.array(), true);
}

// general purpose function which call system command
// and prints output and error streams from the
// launched programm
public void execSystemCommand(String[] commands, boolean printOutput) {
  Runtime rt = Runtime.getRuntime();
  println("calling script : ");
  for(String a : commands) {
    print(a + " ");
  }
  println();
  Process proc = null;
  try {
    proc = rt.exec(commands);
  } catch(Exception e) {
    println(e);
  }
  // print error stream
  if(proc != null) {
    if(printOutput) {
      BufferedReader stdErr = new BufferedReader(new 
           InputStreamReader(proc.getErrorStream()));
           
      // read the output from the command
      String se = null;
      try{
        while ((se = stdErr.readLine()) != null) {
          println(se);
        }
      } catch (Exception e) {
        println(e);
      }
      // print standard stream
      BufferedReader stdout = new BufferedReader(new 
           InputStreamReader(proc.getInputStream()));      
      // read the output from the command
      String so = null;
      try{
        while ((so = stdout.readLine()) != null) {
          println(so);
        }
      } catch (Exception e) {
        println(e);
      }
    }
  }
  else {
    println("ERROR : unable to exec command");
  }
}

// take a looong string and cut it in several strings so that they
// fit in a window of width 'windowWidth' when displayed
// textSize() must have been called in order to this function work
StringList limitTextWidth(String text, float windowWidth) {
  StringList pieces = new StringList();
  for(String line : split(text, '\n')) {
    String piece = "";
    int i = -1;
    for(String word : split(line, ' ')) {
      i++;
      if(textWidth(piece + " " + word) < windowWidth) {
        if(i > 0) piece += " ";
        piece += word;
      }
      else {
        pieces.append(piece);
        piece = word;
      }
    }
    pieces.append(piece);
  }
  return pieces;
}

// function to remove the last piece of sentence in the text
// returns a single string without the last unfinished sentence
String removeLastSentence(String input) {
  String[] terminations = {".", ".'", ".\"", 
                           "?", "?'", "!\"", 
                           "!", "!'", "!\""
                           };
  int lastIndex = 0;
  for(String t : terminations) {
    int terminationIndex = input.lastIndexOf(t);
    if(lastIndex < terminationIndex) {
      lastIndex = terminationIndex + t.length();
    }
  }
  return input.substring(0, lastIndex);
}

void setMode(int newMode) {
  if(newMode < MODE_LANG || newMode > MODE_DISPLAY) {
    println("WARNING : try to set mode with invalid value : " + newMode);
    return;
  }
  if(mode == MODE_EDIT && newMode != MODE_EDIT) {
    // make controls not visible
    println("close gEdit panel");
    gEdit.setPosition(-4000, -4000);
  }
  if(newMode == MODE_EDIT) {
    // make controls visible
    println("open gEdit panel");
    gEdit.setPosition(0, 0);
  }
  if(newMode == MODE_LANG) {
    // hide skip button
    cp5.get(Bang.class, "skip").setPosition(-4000, -2000);
  }
  mode = newMode;
  modeChrono = millis();
}
