function move_slider(id, translation) {
  let slider = document.querySelector(`#${id} .rangeslider-container`);
  if (!slider) return;

  let currentTransform = slider.getAttribute("transform");
  if (!currentTransform) return;

  // Skip if same translation is already applied
  if (move_slider.lastTransform === currentTransform && move_slider.lastTranslation === translation) {
      return;
  }

  let regex = /\(([^)]+)\)/;
  let match = regex.exec(currentTransform);
  if (!match) return;

  let position = match[1].split(",").map(val => parseInt(val));
  let translate = regex.exec(translation)[1].split(",").map(val => parseInt(val));

  let newTransform = `translate(${translate[0] + position[0]}, ${translate[1] + position[1]})`;
  
  if (newTransform !== currentTransform) {
      slider.setAttribute("transform", newTransform);
      move_slider.lastTransform = newTransform;
      move_slider.lastTranslation = translation;
  }
}

function graph_container() {
  console.log("graph_container called");
  var container = document.getElementById('graph-container');
  if (container) {
    container.scrollTop = container.scrollHeight;
  }
}

var graphDiv = document.getElementById('meg-signal-graph');
console.log("graphDiv found:", !!graphDiv);

if (graphDiv) {
  // Try both ways:
  if (graphDiv.on) {
    // Plotly on method
    graphDiv.on('plotly_afterplot', graph_container);
  } else {
    // fallback to DOM event listener
    graphDiv.addEventListener('plotly_afterplot', graph_container);
  }
}


