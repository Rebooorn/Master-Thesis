print("Have fun labeling :)");
print("");
print("Instruction: \n Remember to save original image first, then label and save all masks. ");
print(" Press [o] to save original image \n Press [0] to save label_class_0 \n Press [1] to save label_class_1");
print("");
print(" Press [i] to make inverse \n Press [m] to make mask \n Press [n] to go to next labeling image");

// add additional dir here to expand number of classes
var label_1_dir = "D:\\ChangLiu\\MasterThesis\\Master-Thesis\\TrainSet\\Label_Class_1";
var label_2_dir = "D:\\ChangLiu\\MasterThesis\\Master-Thesis\\TrainSet\\Label_Class_2";
var origin_img_dir = "D:\\ChangLiu\\MasterThesis\\Master-Thesis\\TrainSet\\Origin_img";
var unprocessed_img_dir = "D:\\ChangLiu\\MasterThesis\\Master-Thesis\\TrainSet\\unprocessed";

/*
selectWindow("optic.jpg");
run("Create Mask");


selectWindow("Mask");
saveAs("tiff", "D:\\Mask.tif");
*/

/*
dir = "D:\\ChangLiu\\MasterThesis\\Master-Thesis\\TrainSet\\Origin_img"
list = getFileList(dir);
for (i = 0; i < list.length; i++) {
	print(list[i]);
}
*/

// start from ending index of last time
var ind = getStartingIndex(label_1_dir);

var unprocessed_list = getFileList(unprocessed_img_dir);
var unprocessed_ind = 0;
open(unprocessed_img_dir + "\\" + unprocessed_list[0]);
print(unprocessed_list[unprocessed_ind] + ":");


////////////////////////////////////////////////////////////////////////////////

// helper functions 

////////////////////////////////////////////////////////////////////////////////

function pad (a, left) { 
	// function that pads integers to string, z.B. 1 => 00001
	while (lengthOf(""+a)<left) a="0"+a; 
	return ""+a; 
}

function getStartingIndex(dir){
	// read all files in dir and determine the start index
	list = getFileList(dir);
	if (list.length==0) {
		return 0;
	}
	existing_label = newArray(list.length);
	for (i = 0; i < list.length; i++) {
		ind = parseInt( substring(list[i], 0, 5) );
		existing_label[i] = ind;
	}
	Array.getStatistics(existing_label, min, max, mean, std);
	return max+1;
}

function getSaveFilename (ind){
	path = pad(ind, 5);
	path = path + ".jpg";
	return path;
}

////////////////////////////////////////////////////////////////////////////////

// Keyboard shortcuts

////////////////////////////////////////////////////////////////////////////////


macro "Nothing" {
	
}

macro "original image [o]" {
	// save mask image to original dir
	print("original image saved");
	print("Save as: " + getSaveFilename(ind));
	saveAs("Jpeg", origin_img_dir + "\\" + getSaveFilename(ind));
	ind = ind + 1;
}

macro "label class 1 [1]" {
	// save mask image to label 1 dir
	print("label class 1 saved");
	saveAs("Jpeg", label_1_dir + "\\" + getSaveFilename(ind-1));
}

macro "label class 2 [2]" {
	// save mask image to label 0 dir
	print("label class 2 saved");
	saveAs("Jpeg", label_2_dir + "\\" + getSaveFilename(ind-1));
}

macro "create mask [m]" {
	print("create mask of selected region");
	run("Create Mask");
}

macro "next image [n]" {
	// close this image and open next image
	close();
	unprocessed_ind = unprocessed_ind + 1;
	if (unprocessed_ind < unprocessed_list.length){
		//print(unprocessed_img_dir + "\\" + unprocessed_list[unprocessed_ind]);
		open(unprocessed_img_dir + "\\" + unprocessed_list[unprocessed_ind]);
		print("////////////////////////////////////");
		print(unprocessed_list[unprocessed_ind] + ":");
	}
	else {
		print("no next image, all done");
	}
}

macro "make inverse [i]" {
	print("make inverse");
	run("Make Inverse");
}



