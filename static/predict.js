let imageLoaded = false;
$("#image-selector").change(function () {
	imageLoaded = false;
	let reader = new FileReader();
	reader.onload = function () {
		let dataURL = reader.result;
		$("#selected-image").attr("src", dataURL);
		$("#prediction-list").empty();
		imageLoaded = true;
	}
	
	let file = $("#image-selector").prop('files')[0];
	reader.readAsDataURL(file);
});

let model;
let modelLoaded = false;
$( document ).ready(async function () {
	modelLoaded = false;
	$('.progress-bar').show();
    console.log( "Loading model..." );
    model = await tf.loadGraphModel('model/IA-v2_tfjs/model.json');
    console.log( "Model loaded." );
	$('.progress-bar').hide();
	modelLoaded = true;
});

$("#predict-button").click(async function () {
	if (!modelLoaded) { alert("The model must be loaded first"); return; }
	if (!imageLoaded) { alert("Please select an image first"); return; }
	
	let image = $('#selected-image').get(0);

	const preprocessImageToTensor = (image) => {
		try {
		  // Convert the image to a tensor
		  const imgTensor = tf.browser.fromPixels(image);
	
		  // Resize the image to the expected size
		  const resizedTensor = tf.image.resizeBilinear(imgTensor, [224, 224]).toFloat();
	  
		  // Normalize the image
		  const mean = [0.485, 0.456, 0.406];
		  const std = [0.229, 0.224, 0.225];
		  const normalizedTensor = resizedTensor.div(tf.scalar(255.0))
			.sub(tf.tensor(mean))
			.div(tf.tensor(std));
	  
		  // Add a batch dimension
		  const batchedTensor = normalizedTensor.expandDims(0);
	  
		  return batchedTensor;
		} catch (error) {
		  console.error("Error transforming image to tensor: ", error);
		  throw new Error("Error during image preprocessing");
		}
	};
	
	// Pre-process the image
	console.log( "Loading image..." );
	let tensor = preprocessImageToTensor(image);
	let predictions = await model.predict(tensor).array();
	console.log(predictions[0]);
	let top8  = predictions[0].map((score, index) => {
		return {
		  score: score,
		  className: TARGET_CLASSES[index] // Sélection de la valeur depuis l'objet
		};
		}).sort(function (a, b) {
			return b.score - a.score;
		}).slice(0, 7);

	$("#prediction-list").empty();
	top8.forEach(function (p) {
		console.log(p.className + ": "  + p.score);
		$("#prediction-list").append(`<li>${p.className}: ${p.score}</li>`);
		});
});
