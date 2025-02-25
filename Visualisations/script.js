// Set the dimensions and margins of the graph
var margin = { top: 50, right: 30, bottom: 70, left: 70 },
    width  = 800 - margin.left - margin.right,
    height = 600 - margin.top - margin.bottom;

// Append an SVG object to the div called 'my_dataviz'
var svg = d3.select("#my_dataviz")
  .append("svg")
    .attr("width",  width  + margin.left + margin.right)
    .attr("height", height + margin.top  + margin.bottom)
  .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

// Load the CSV data (make sure data.csv is in the same folder)
d3.csv("experiment_results.csv", function(error, data) {
    if (error) {
      console.error("Error loading data: ", error);
      return;
    }
    console.log("All column names in first row:", Object.keys(data[0]));
    for (let i = 5; i < 10; i++) {
        console.log(`Row ${i} Detected_Emotion:`, data[i].Detected_Emotion);
      }
      

  // --- Step 1: Preprocess the Data ---
  // For each row, split the Detected_Emotion column into individual emotions
  var flattenedData = [];
  data.forEach(function(d) {
    // Get the team for the row
    var clipType = d.Clip_Type;
    // Split the Detected_Emotion string into an array (assuming semicolon-separated values)
    var emotions = d.Detected_Emotion.split(";").map(function(emotion) { 
      return emotion.trim(); 
    });
    // For each emotion, push a new record into flattenedData
    emotions.forEach(function(emotion) {
      flattenedData.push({ Clip_Type: clipType, Detected_Emotion: emotion });
    });
  });

  // --- Step 2: Group with d3.nest() and roll up counts ---
  // d3.nest() in D3 v4 returns an array of objects, each with `key` and `values`.
  var countsMap = d3.nest()
    .key(function(d) { return d.Clip_Type; })
    .key(function(d) { return d.Detected_Emotion; })
    .rollup(function(values) {
      return values.length; // the number of items in this group
    })
    .entries(flattenedData);

  // --- Step 3: Convert the nested structure into a flat array ---
  // Each object in countsMap looks like:
  // { key: "TN", values: [ { key: "Anger", value: 3 }, { key: "Fear", value: 2 }, ... ] }
  var counts = [];
  countsMap.forEach(function(clipObj) {
    clipObj.values.forEach(function(emotionObj) {
      counts.push({
        Clip_Type: clipObj.key,
        Detected_Emotion: emotionObj.key,
        Count: emotionObj.value
      });
    });
  });


// 'counts' is now an array like:
// [
//   { Clip_Type: "TN", Detected_Emotion: "Anger", Count: 3 },
//   { Clip_Type: "TN", Detected_Emotion: "Fear",  Count: 2 },
//   ...
// ]


//   // Convert the Count field to a number
//   data.forEach(function(d) {
//     d.Count = +d.Count;
//   });

  // Create an x-scale: a band scale for the 'Team' (categorical)
  var x = d3.scaleBand()
    .domain(counts.map(function(d) { return d.Clip_Type; }))
    .range([0, width])
    .padding(1);

  // Create a y-scale: a band scale for 'Emotion' (categorical)
  var y = d3.scaleBand()
    .domain(counts.map(function(d) { return d.Detected_Emotion; }))
    .range([height, 0])
    .padding(1);

  // Create a scale for bubble size based on 'Count'
  var r = d3.scaleLinear()
    .domain([0, d3.max(counts, function(d) { return d.Count; })])
    .range([0, 30]); // Adjust max radius as needed

  // Create an ordinal color scale for 'Emotion'
  var color = d3.scaleOrdinal(d3.schemeSet2);

  // Add the X axis to the SVG
  svg.append("g")
    .attr("transform", "translate(0," + height + ")")
    .call(d3.axisBottom(x));

  // Add the Y axis to the SVG
  svg.append("g")
    .call(d3.axisLeft(y));

  // Create the bubbles
  svg.selectAll("circle")
    .data(counts)
    .enter()
    .append("circle")
      .attr("cx", function(d) { return x(d.Clip_Type); })
      .attr("cy", function(d) { return y(d.Detected_Emotion); })
      .attr("r",  function(d) { return r(d.Count); })
      .style("fill", function(d) { return color(d.Detected_Emotion); })
      .style("opacity", 0.7)
      .attr("stroke", "white")
      .style("stroke-width", "2px");

  console.log(counts);

  if (error) {
    console.error("Error loading data:", error);
    return;
  }

});
