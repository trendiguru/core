const [x1, y1, x2, y2] = [440, 80, 640, 280];
let img = document.querySelector("img");
const API_URL = `http://13.82.136.127:8082/hls?x1=${x1}&x2=${x2}&y1=${y1}&y2=${y2}&imageUrl=${img.src}`;
let container = document.createElement("DIV");
container.appendChild(img);
Object.assign(container.style, {
  position: "relative",
  border: "2px solid green"
});
document.body.appendChild(container);
let roiBox = createBox({bbox: [x1, y1, x2, y2]});
roiBox.style.border = "1px solid red";
container.appendChild(roiBox);

fetch(`${API_URL}`)
  .then((response) => response.json())
  .then((rjson) => rjson.data)
  .then(drawBoxes);


function createBox({bbox, object, confidence}) {
  let boxDiv = document.createElement("DIV");
  Object.assign(boxDiv.style, {
    position: "absolute",
    border: "1px solid blue",
    zindex: 100,
    left: `${bbox[0]}px`,
    top: `${bbox[1]}px`,
    width: `${bbox[2]-bbox[0]}px`,
    height: `${bbox[3]-bbox[1]}px`
  });
  return boxDiv;
}

function drawBoxes(data){
	for (let obj of data) {
  console.log(obj);
  boxDiv = createBox(obj);
  container.appendChild(boxDiv);
}
}