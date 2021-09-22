import netP5.*;
import oscP5.*;

import controlP5.*;


final int MODE_LANG = 0;
final int MODE_EDIT = 1;
final int MODE_GENERATE = 2;
final int MODE_DISPLAY = 3;

OscP5 oscP5;
int listeningPort = 9000;
boolean handshake = false;

ControlP5 cp5;
Group gEdit;

String generatorScript = "D:\\code\\send_analyze_osc.py";
String modelsDirectory = "D:\\models";
String[] modelNames;
int modelIndex = 0;
String primeText = " ";
int nbWords = 180;

String result = "";
StringList text;

PGraphics pg;
int textSize = 40;
float y0 = 0;
float xOffset = 60;
float speed = 0.5;

int mode;
int modeChrono = 0;

void setup() {
  size(1280, 720);
  //fullScreen(P3D);
  
  pg = createGraphics(int(width), int(height * 1.25));
  
  oscP5 = new OscP5(this, listeningPort);
  
  textSize = width / 24;
  println("text size set to " + textSize);
  
  // list models from given directory
  println("Listing all models from " + modelsDirectory + " : ");
  modelNames = listFileNames(modelsDirectory);
  printArray(modelNames);
  
  text = new StringList();
  
  // GUI
  CColor customGuiColor = ControlP5.THEME_GREY;
  customGuiColor.setCaptionLabel(color(0));
  customGuiColor.setValueLabel(color(0));
  
  cp5 = new ControlP5(this);
  gEdit = cp5.addGroup("edit")
    .setPosition(0,0)
    .setSize(width, height)
    .setBackgroundHeight(height)
    .setBackgroundColor(color(32))
    ;
  // dropdown to choose the model
  StringList cleanedModelNames = new StringList();
  for(String n : modelNames) {
    String nCleaned = n.replaceAll("_", " ").replaceAll("-", " ").substring(0, n.length() - 3);
    cleanedModelNames.append(nCleaned);
  }
  cp5.addScrollableList("model")
     .setGroup(gEdit)
     .setPosition(40, 40)
     .setSize(textSize * 8, height - 40 * 4 - textSize * 4)
     .setColor(customGuiColor)
     .setFont(createFont("arial", textSize * 0.3))
     .setBarHeight(round(textSize * 0.6))
     .setItemHeight(round(textSize * 0.6))
     .addItems(cleanedModelNames.array())
     .setType(ScrollableList.DROPDOWN)
     .setValue(0)
     ;
  // dropdown to choose generated text length
  String[] textLengthChoice = {"Short", "Medium", "Long", "longer", "Longest"};
  cp5.addScrollableList("textLengthChoice")
     .setGroup(gEdit)
     .setPosition(width - 40 - textSize * 8, 40)
     .setSize(textSize * 8, height - 40 * 4 - textSize * 4)
     .setColor(customGuiColor)
     .setFont(createFont("arial", textSize * 0.3))
     .setBarHeight(round(textSize * 0.6))
     .setItemHeight(round(textSize * 0.6))
     .addItems(textLengthChoice)
     .setType(ScrollableList.DROPDOWN)
     .setValue(1)
     ;
  // text input field for prime text
  cp5.addTextfield("textSeed")
     .setGroup(gEdit)
     .setPosition(40, height - 40 * 3 - textSize * 4)
     .setSize(width - 80, textSize)
     .setColor(customGuiColor)
     .setFont(createFont("arial", textSize * 0.6))
     .setFocus(true)
     .setAutoClear(false)
     .getCaptionLabel().setVisible(false)
     ;
  cp5.addBang("clear")
     .setGroup(gEdit)
     .setPosition(width - textSize * 2 - 40, height - 40 * 2 - textSize * 3)
     .setSize(textSize * 2, textSize)
     .setTriggerEvent(Bang.RELEASE)
     .setColor(customGuiColor)
     .setFont(createFont("arial", textSize * 0.3))
     .getCaptionLabel().align(ControlP5.CENTER, ControlP5.CENTER)
     ;    
  // bang button to launch generation
  cp5.addBang("generate")
     .setGroup(gEdit)
     .setPosition(40, height - 40 - textSize * 2)
     .setSize(width - 80, textSize * 2)
     .setTriggerEvent(Bang.PRESS)
     //.setColor(ControlP5.THEME_RED)
     .setFont(createFont("arial", textSize * 1.0))
     .getCaptionLabel().align(ControlP5.CENTER, ControlP5.CENTER)
     ;
  // not visible at the begining
  gEdit.setPosition(-4000, -4000);
  
  // skip button to skip text display if too long
  CColor skipColor = ControlP5.THEME_GREY;
  skipColor.setCaptionLabel(color(220));
  skipColor.setBackground(color(220));
  skipColor.setForeground(color(64));
  skipColor.setActive(color(150));
  cp5.addBang("skip")
     .setPosition(width - 20 - 2 * textSize, 20)
     .setSize(round(textSize), round(textSize * 0.6))
     .setTriggerEvent(Bang.PRESS)
     .setColor(skipColor)
     .setFont(createFont("arial", textSize * 0.3))
     .getCaptionLabel().align(ControlP5.CENTER, ControlP5.CENTER)
     ;
  // not visible at the begining
  cp5.get(Bang.class, "skip").setPosition(-4000, -2000);
     
  imageMode(CENTER);
  textSize(textSize);
  fill(255);
  textAlign(CENTER);
  
  pg.beginDraw();
  pg.textAlign(LEFT);
  pg.textSize(textSize);
  pg.strokeWeight(1);
  pg.endDraw();
  
  setMode(MODE_LANG);
}


void draw() {
  if(mode == MODE_LANG) {
    setMode(MODE_EDIT);
    background(32);
  }
  if(mode == MODE_EDIT) {
    background(32);
    Textfield textSeedRef = cp5.get(Textfield.class,"textSeed");
    if(textSeedRef.isFocus()) {
      strokeWeight(4);
      stroke(25, 192, 243);
      rect(textSeedRef.getPosition()[0], textSeedRef.getPosition()[1], 
            textSeedRef.getWidth(), textSeedRef.getHeight());
    }
  }
  else if(mode == MODE_GENERATE) {
    background(32);
    translate(width/2, height/2);
    strokeWeight(8);
    rotate(radians(frameCount / 15 * 30));
    for(int i = 0; i < 12; i++) {
      stroke(255.0 - (i * 256.0) / 12);
      line(80, 0, 280, 0);
      rotate(radians(-30));
    }
    
  }
  else if(mode == MODE_DISPLAY) {
    background(32);
    pg.beginDraw();
    pg.background(32);
    float y = y0 + 20;
    synchronized(text) {
      pg.fill(255, 255, 127);
      for(String line : text) {
        pg.text(line, xOffset , y);
        y += textSize * 1.2;
      }
    }
    // make the text disappear smoothly at the top of the PGraphics
    for(int i=0; i < textSize * 3; i++) {
      pg.stroke(32, 255.0 * ((textSize * 3) - i) / (textSize * 3.0));
      pg.line(0, i, pg.width, i);
    }
    pg.endDraw();
    pushMatrix();
    translate(width/2, height/2, -200);
    rotateX(radians(45));
    image(pg, 0, 0);
    popMatrix();
    
    y0 -= speed;
    
    if(y < 0) {
      setMode(MODE_LANG);
    }
    
    // propose to skip text display after 15 seconds
    if(millis() - modeChrono > 15000 && cp5.get(Bang.class, "skip").getPosition()[0] < 0 ) {
      println("show skip bang");
      cp5.get(Bang.class, "skip").setPosition(width - 20 - 2 * textSize, 20);
    }
  }
}



void oscEvent(OscMessage msg) {
  if(msg.checkAddrPattern("/multi_rnn_wordlevel/handshake")) {
    println("handshake received!");
    handshake = true;
  }
  if(msg.checkAddrPattern("/multi_rnn_wordlevel/result")) {
    // get current packet index and total number of packets for this result
    int packet = msg.get(0).intValue();
    int totalPackets = msg.get(1).intValue();
    // append new packet string to 'result'
    result += msg.get(2).stringValue();
    if(packet == 1) {
      // fisrt packet, set mode to display
      setMode(MODE_DISPLAY);
    }
    if(packet == totalPackets) {
      // last packet received, print and process result and clear 'result' string
      //println("result received : \n" + result);
      println("result received !");
      char[] resultArray = result.toCharArray();
      for(int i = 0; i < resultArray.length; i++) {
        char c = resultArray[i];
        if((c & 0xC0) == 0xC0) {
          resultArray[i] = (char) ((c & 0x001F) << 6 | resultArray[i+1] & 0x003F);
          resultArray[i+1] = '\0';
        }
      }
      result = new String(resultArray).replaceAll("\0", "");
      //result = removeLastSentence(result);
      //println("result cleaned : \n" + result);
      handshake = false;
      y0 = pg.height;
      synchronized(text) {
        text.clear();
        text.append(limitTextWidth(result, pg.width - 2 * xOffset));
      }
      result = "";
    }
  }
}


////// GUI Callbacks /////
public void model(int n) {
  modelIndex = n;
  println("Set corpus model to " + modelNames[modelIndex]);
}

public void textLengthChoice(int n) {
  int nb = n + 1;
  nbWords = 80 + nb * nb * 20;
  println("Set number of words to " + nbWords);
}

public void clear() {
  cp5.get(Textfield.class,"textSeed").clear();
}

public void generate() {
  if(mode == MODE_EDIT) {
    primeText = cp5.get(Textfield.class,"textSeed").getText();
    cp5.get(Textfield.class,"textSeed").clear();
    println("Set prime text to : " + primeText);
    if(primeText.length() == 0) primeText = " ";
    setMode(MODE_GENERATE);
    //String[] command = {"python", "--version"};
    //execSystemCommand(command);
    thread("execScriptCommand");
  }
}

public void skip() {
  println("skip text display");
  setMode(MODE_LANG);
}
