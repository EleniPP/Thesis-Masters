// Variables
let canvas, stream, recorder, chunks = [];
let recordingDuration = 8500; // 3.5 seconds

// Function to extract filename without extension
function getFileNameWithoutExtension(fileInput) {
    if (!fileInput || fileInput.files.length === 0) {
        return "output"; // Default name if no file is selected
    }
    const fileName = fileInput.files[0].name; // Get full filename (e.g., "patient_123.txt")
    return fileName.replace(/\.[^/.]+$/, ""); // Remove the extension (".txt")
}

// Function to start recording when "Absolute" checkbox is checked
function startRecording() {
    if (!canvas) {
        canvas = document.querySelector("canvas");
    }

    stream = canvas.captureStream(30); // 30 FPS
    // Capture the audio from the audio player
    audioStream = audioPlayer.captureStream(); // Captures the currently playing audio
    mixedStream = new MediaStream([...stream.getTracks(), ...audioStream.getTracks()]); // Combine the video and audio streams

    recorder = new MediaRecorder(mixedStream, { mimeType: 'video/webm; codecs=vp9' });

    chunks = []; // 🚀 Reset chunks BEFORE starting new recording

    recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            chunks.push(event.data);
        }
    };

    recorder.onstop = async () => {
        console.log("Recording stopped, saving video...");
        await saveVideo();
    };

    recorder.start();
    setTimeout(() => {
        if (recorder.state === "recording") {
            recorder.stop();
        }
    }, recordingDuration);
}

// Function to save the recorded video
function saveVideo() {
    const fileNameWithoutExt = getFileNameWithoutExtension(document.getElementById('file')); // Get filename from input
    const blob = new Blob(chunks, { type: 'video/webm' });
    const url = URL.createObjectURL(blob);

    // Create a download link
    const a = document.createElement('a');
    a.href = url;
    a.download = `${fileNameWithoutExt}.webm`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    console.log("Video saved as .webm!");
}

// Audio Player
const audioFileInput = document.getElementById('audioFileInput');
// const absoluteCheckbox = document.getElementById("absoluteCheckbox");

// Create an audio player dynamically
const audioPlayer = new Audio();


// When the user selects an audio file, update the audio player source
audioFileInput.addEventListener('change', (event) => {
    const file = event.target.files[0]; // Get the selected file
    if (file) {
        const objectURL = URL.createObjectURL(file); // Create a temporary URL for the file
        audioPlayer.src = objectURL; // Set the audio player's source to the selected file
		// playAudioButton.disabled = false; // Enable the Play button
        console.log(`Audio file selected: ${file.name}`);
    }
});

// Variables to control rendering
let renderingStarted = false; // Rendering starts only when the checkbox is clicked

// Event listener for the absolute checkbox
document.getElementById("absoluteCheckbox").addEventListener("change", (event) => {
    if (event.target.checked) {
        // Start rendering only if not already started
        if (!renderingStarted) {
			// Reset faceDataListIndex to 0
			faceDataListIndex = 0;

			// Reset audio playback to the beginning
			audioPlayer.currentTime = 0;
            renderingStarted = true; // Mark rendering as started
            window.requestAnimationFrame(renderFunction); // Start the rendering loop
            console.log("Rendering started!");

			// Automatically play the audio
			audioPlayer.play()
				.then(() => console.log("Audio playback started"))
				.catch((error) => console.error("Audio playback failed:", error));

            // Start automatic recording
            startRecording();
        }
    }
});

function renderFaceData(faceData)
{
	let personIndex = 0;
    context.lineWidth = 1;
    console.log("Rendering face data. Frame " + faceDataListIndex);
    
    if (faceData == null)
        return;
    
    if ((typeof faceData) == "string")
        faceData = JSON.parse(faceData);
	
	let globalConfidence = faceData["confidence"]
	let points = []
	
	for (var pointIndex = 0; pointIndex < 68; pointIndex++)
	{
		var x = parseInt(faceData["x" + pointIndex]);
		var y = parseInt(faceData["y" + pointIndex]);
		
		points.push({x: x, y: y, c: globalConfidence, radius: null, extremeX: null, extremeY: null});
	}
        
	context.fillStyle = 'hsl(185, 0%, 63%)';
	context.strokeStyle = 'black';
	
	let minX = points.map(e => e.x).sort(e => e)[0]
	let maxX = points.map(e => e.x).sort(e => -e)[0]
	let minY = points.map(e => e.y).sort(e => e)[0]
	let maxY = points.map(e => e.y).sort(e => -e)[0]
	
	console.log("\t(" + minX + ", "  + maxX + ", "  + minY + ", "  + maxY + ")")
	console.log("\tConf: " + globalConfidence)


	// Draw confidence ellipses around face points.
	for (var pointIndex = 0; pointIndex < points.length; pointIndex += 1)
	{
		point = points[pointIndex];
		
		if (point.c < confidenceLimit)
			continue;
		
		var normalisedX = absoluteRendering ? point.x : canvas.width * (point.x - minX) / (maxX - minX);
		var normalisedY = absoluteRendering ? point.y : canvas.height * (point.y - minY) / (maxY - minY);
		
		context.beginPath();
		var dX = point.extremeX - point.x;
		var radiusX = Math.abs(dX);
		var dY = point.extremeY - point.y;
		var radiusY = Math.abs(dY);
		var averageX = (point.extremeX + point.x) / 2;
		var averageY = (point.extremeY + point.y) / 2;
		var ellipseRotation = Math.atan2(dY, dX);
		/*var radius = (point.radius != null) ? point.radius : 4
		context.arc(normalisedX, normalisedY, radius, 0, 2*Math.PI);*/
		context.ellipse(averageX, averageY, radiusX, radiusY, ellipseRotation, 0, 2*Math.PI);
		// context.globalAlpha = Math.max(0, Math.min(1, 4 / (radius)));
		context.globalAlpha = Math.max(0, Math.min(1, 2 / (radiusX + radiusY)));
		context.fill();
		context.stroke();
	}
        
	// Draw face outline.
	let drawOutline = true; // disabled
	if (absoluteRendering && drawOutline)
	{
		context.beginPath();
		context.moveTo(points[0].x, points[0].y);
		for (var outlinePoint = 1; outlinePoint < 17; outlinePoint += 1)
		{
			context.lineTo(points[outlinePoint].x, points[outlinePoint].y);
		}
		for (var outlinePoint = 26; outlinePoint >= 17; outlinePoint -= 1)
		{
			context.lineTo(points[outlinePoint].x, points[outlinePoint].y);
		}
		context.fillStyle = 'rgba(216, 216, 216, 0.74)';
		context.fill();
	}
	
	// Draw actual face points.
	for (var pointIndex = 0; pointIndex < points.length; pointIndex += 1)
	{
		point = points[pointIndex];
		
		var normalisedX = absoluteRendering ? point.x : canvas.width * (point.x - minX) / (maxX - minX);
		var normalisedY = absoluteRendering ? point.y : canvas.height * (point.y - minY) / (maxY - minY);
		
		
		if (point.extremeX != null && point.extremeY != null)
		{
			var extremeX = absoluteRendering ? point.extremeX : canvas.width * (extremeX - minX) / (maxX - minX);
			var extremeY = absoluteRendering ? point.extremeY : canvas.height * (extremeY - minY) / (maxY - minY);
			
			context.beginPath();
			context.moveTo(extremeX, extremeY);
			context.lineTo(normalisedX, normalisedY);
			context.stroke();
		}
		
		context.beginPath();
		
		context.arc(normalisedX, normalisedY, 1 + 1*point.c, 0, 2*Math.PI);
	
		var transparency = point.c < confidenceLimit ? (point.c / confidenceLimit) : 1;
		context.fillStyle = `hsla(${personIndex * 45}, ${100*point.c}%, 50%, ${transparency})`;
		context.fill();
		context.stroke();
	}
}

function parseCSV(str) {
    var arr = [];
    var quote = false;
    for (var row = col = c = 0; c < str.length; c++) {
        var cc = str[c], nc = str[c+1];
        arr[row] = arr[row] || [];
        arr[row][col] = arr[row][col] || '';

        if (cc == '"' && quote && nc == '"') { arr[row][col] += cc; ++c; continue; }  
        if (cc == '"') { quote = !quote; continue; }
        if (cc == ',' && !quote) { ++col; continue; }
        if (cc == '\r' && nc == '\n' && !quote) { ++row; col = 0; ++c; continue; }
        if (cc == '\n' && !quote) { ++row; col = 0; continue; }
        if (cc == '\r' && !quote) { ++row; col = 0; continue; }

        arr[row][col] += cc;
    }
    return arr;
}

function parseCSVtoObjects(csvString, /* optional */ columnNames) {
    var csvRows = parseCSV(csvString);

    var firstDataRow = 0;
    if (!columnNames) {
        columnNames = csvRows[0];
        firstDataRow = 1;
    }

    var result = [];
    for (var i = firstDataRow, n = csvRows.length; i < n; i++) {
        var rowObject = {};
        var row = csvRows[i];
        for (var j = 0, m = Math.min(row.length, columnNames.length); j < m; j++) {
            var columnName = columnNames[j].trim();
            var columnValue = row[j];
            rowObject[columnName] = columnValue;
        }
        result.push(rowObject);
    }
    return result;
}

function loadFaceDataCSV(data)
{
	var objects = parseCSVtoObjects(data);
	window.faceDataListIndex = 0
	window.faceDataList = objects;
}

function loadFaceDataList(data)
{
    faceDataList = JSON.parse(data);
    faceDataListIndex = 0;
}

function syncFaceToAudio() {
    const currentTime = audioPlayer.currentTime; // Convert seconds to minutes
	// const faceDataStartTime = faceDataList[0]?.timestamp || 0; // Get first timestamp (assumed to be the start time)
	// const adjustedAudioTime = audioPlayer.currentTime + faceDataStartTime; // Align with face data timeline
    // Find the closest frame to the current audio time
    for (let i = faceDataListIndex; i < faceDataList.length; i++) {
        if (parseFloat(faceDataList[i].timestamp) >= currentTime) {
            faceDataListIndex = i; // Sync frame index
            faceData = faceDataList[i];
            break;
        }
    }

	console.log(`Print audio and frame timestamp`);
    console.log(`Audio timestamp: ${currentTime} seconds`);
    console.log(`Frame timestamp: ${faceDataList[faceDataListIndex]?.timestamp} seconds`);
}

let lastRenderTime = 0;
function renderFunction(timestamp)
{   
	if (!lastRenderTime) lastRenderTime = timestamp;

    // Calculate time elapsed since the last frame
    const timeElapsed = timestamp - lastRenderTime;

    // Only render a new frame every 33ms (30 FPS)
    if (timeElapsed >= 33) {
		syncFaceToAudio(); // Synchronize frame with audio
		let alwaysClear = true;

		if (alwaysClear)
			canvas.width |= 0;

		if (canvas.width != window.innerWidth || canvas.height != window.innerHeight)
		{
			canvas.width = window.innerWidth;
			canvas.height = window.innerHeight;
			
			window.lastFaceData = null;
		}
		if (lastFaceData == undefined || faceData != lastFaceData)
		{
			context.fillStyle = `hsl(0, 50%, 100%)`;
			context.fillRect(0, 0, canvas.width, canvas.height);
			renderFaceData(faceData);
			lastFaceData = faceData;
		}
		
		if (loop == true && faceDataList != null && faceDataListIndex >= faceDataList.length)
		{
			faceDataListIndex = 0;
		}
		
		if (faceDataList != null && faceDataListIndex < faceDataList.length)
		{
			faceData = faceDataList[faceDataListIndex];
			faceDataListIndex += 1;
			
			var frameString = faceDataListIndex + " / " + faceDataList.length;
			
			context.fillStyle = 'black';
			context.fillText(frameString, 5, 5);
		}
		lastRenderTime = timestamp; // Update the last render time
	}
	// Loop the rendering if the checkbox remains checked
	if (renderingStarted) {
		window.requestAnimationFrame(renderFunction);
	}
    // window.requestAnimationFrame(renderFunction);
}

function loadDataFromPasteArea()
{
    var data = document.getElementById("pasteArea").value;
    loadFaceDataList(data);
}

document.getElementById('file').onchange = function()
{
  var file = this.files[0];

  var reader = new FileReader();
  reader.onload = function(progressEvent)
  {
    loadFaceDataCSV(this.result);
  };
  reader.readAsText(file);
};

document.getElementById("smoothingSlider").oninput = function()
{
    smoothingDistance = this.value;
}

document.getElementById("confidenceSlider").oninput = function()
{
    confidenceLimit = this.value / 100;
}

document.getElementById("absoluteCheckbox").oninput = function()
{
    absoluteRendering = this.checked;
}

canvas = document.getElementById("canvas");
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
context = canvas.getContext("2d");

// Ensure audio doesn't play before data is loaded
audioPlayer.pause();

faceData = null;
lastFaceData = null;

lastPoints = new Map();
lastBones = new Map();
smoothingDistance = 0;
confidenceLimit = 0;
absoluteRendering = false;

boneMap = new Map();
boneMap.set(17, 15);
boneMap.set(18, 16);
boneMap.set(15, 0);
boneMap.set(16, 0);
boneMap.set(0, 1);
boneMap.set(1, 8);
boneMap.set(2, 1);
boneMap.set(5, 1);
boneMap.set(3, 2);
boneMap.set(6, 5);
boneMap.set(4, 3);
boneMap.set(7, 6);
boneMap.set(4, 3);
boneMap.set(9, 8);
boneMap.set(12, 8);
boneMap.set(10, 9);
boneMap.set(13, 12);
boneMap.set(11, 10);
boneMap.set(14, 13);
// Foot bones skipped for brevity.

loop = true;

faceDataList = null;
faceDataListIndex = 0;

// window.requestAnimationFrame(renderFunction);