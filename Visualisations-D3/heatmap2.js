// heatmap.js

// Set margins and dimensions for the SVG container
var margin = { top: 80, right: 80, bottom: 80, left: 200 },
    width = 800 - margin.left - margin.right,
    height = 700 - margin.top - margin.bottom;

// Append an SVG to the #my_dataviz div
var svg = d3.select("#my_dataviz")
  .append("svg")
  .attr("width", width + margin.left + margin.right)
  .attr("height", height + margin.top + margin.bottom)
  .append("g")
  .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

// Function to determine if a classification is correct
function isCorrectClassification(clipType, confidenceLevel) {
  var positiveTypes = ["TP", "FN"]; // True Positive or False Negative (assume depressed)
  var negativeTypes = ["TN", "FP"]; // True Negative or False Positive (assume not depressed)
  var likely = ["Somewhat Likely", "Very Likely"];
  var unlikely = ["Somewhat Unlikely", "Very Unlikely"];

  if (positiveTypes.indexOf(clipType) > -1 && likely.indexOf(confidenceLevel) > -1) {
    return "Correct";
  } else if (negativeTypes.indexOf(clipType) > -1 && unlikely.indexOf(confidenceLevel) > -1) {
    return "Correct";
  } else {
    return "Incorrect";
  }
}

// Function to determine if the ground truth is depressed or not
function getDepressionStatus(clipType) {
    // "TP" (true positive) and "FN" (false negative) mean the person *is* depressed
    // "TN" (true negative) and "FP" (false positive) mean the person is *not* depressed
    if (clipType === "TP" || clipType === "FN") {
      return "Depressed";
    } else {
      return "Not Depressed";
    }
  }

// Load the CSV data
d3.csv("experiment_results-salient_segments.csv", function(error, data) {
  if (error) {
    console.error("Error loading data:", error);
    return;
  }
  
  // Flatten data:
  // For each row, determine classification correctness and split each of the three facial-feature columns.
  var flattenedData = [];
  data.forEach(function(d) {
    // Determine correct or incorrect classification
    // var classification = isCorrectClassification(d.Clip_Type, d.Confidence_Level);
    var status = getDepressionStatus(d.Clip_Type);

//     // for each facial feature
    // For each facial feature column, split the values (assuming semicolon-separated), trim them, and push into flattenedData.
//     var eyebrows = d["Influential_Features-Eyebrows"] ? d["Influential_Features-Eyebrows"].split(";") : [];
//     var eyes = d["Influential_Features-Eyes"] ? d["Influential_Features-Eyes"].split(";") : [];
//     var mouth = d["Influential_Features-Mouth"] ? d["Influential_Features-Mouth"].split(";") : [];
    
//     eyebrows.forEach(function(feat) {
//       feat = feat.trim();
//       if (feat !== "") {
//         flattenedData.push({ status: status, feature: feat });
//       }
//     });
//     eyes.forEach(function(feat) {
//       feat = feat.trim();
//       if (feat !== "") {
//         flattenedData.push({ status: status, feature: feat });
//       }
//     });
//     mouth.forEach(function(feat) {
//       feat = feat.trim();
//       if (feat !== "") {
//         flattenedData.push({ status: status, feature: feat });
//       }
//     });
//   });
  
    // voice features
        // Split the voice features (assuming semicolon-separated)
        var voiceFeatures = d["Influential_Features-Voice"] 
        ? d["Influential_Features-Voice"].split(";") 
        : [];
  
      voiceFeatures.forEach(function(feat) {
        feat = feat.trim();
        if (feat !== "") {
          flattenedData.push({ status: status, feature: feat });
        }
      });
    });

  // Group and count by classification and feature using d3.nest (D3 v4)
  var nested = d3.nest()
    .key(function(d) { return d.status; })
    .key(function(d) { return d.feature; })
    .rollup(function(leaves) { return leaves.length; })
    .entries(flattenedData);
  
  // Convert the nested structure into a flat array:
  // Each object will have { classification, feature, count }
  var counts = [];
  nested.forEach(function(statusObj) {
    statusObj.values.forEach(function(featureObj) {
      counts.push({
        status: statusObj.key,
        feature: featureObj.key,
        count: featureObj.value
      });
    });
  });
  
  // Define the order for classification on the x-axis
//   var classifications = ["Incorrect", "Correct"];
  var statuses = ["Depressed", "Not Depressed"];
  
  // Create a sorted list of all unique features (y-axis)
  var allFeatures = d3.set(counts.map(function(d) { return d.feature; })).values();
  allFeatures.sort();  // sort alphabetically
  
  // Build the X scale for classification (band scale)
  var x = d3.scaleBand()
    .range([0, width])
    .domain(statuses)
    .padding(0.05);
  
  svg.append("g")
    .attr("transform", "translate(0," + height + ")")
    .call(d3.axisBottom(x).tickSize(0))
    .select(".domain").remove();
  
  // Build the Y scale for facial features
  var y = d3.scaleBand()
    .range([height, 0])
    .domain(allFeatures)
    .padding(0.05);
  
  svg.append("g")
    .call(d3.axisLeft(y).tickSize(0))
    .select(".domain").remove();
  
  // Determine maximum count for color scaling
  var maxCount = d3.max(counts, function(d) { return d.count; });
  
  // Build a color scale with your original colors (from light to dark)
  var colorScale = d3.scaleLinear()
    .domain([0, maxCount])
    // .range(["#ffffe0", "#b22222"]);
    .range(["#eff3ff", "#08519c"]);  // light blue to dark blue
  
  // Tooltip for interactivity
  var tooltip = d3.select("body")
    .append("div")
    .attr("class", "tooltip")
    .style("position", "absolute")
    .style("padding", "5px")
    .style("background-color", "white")
    .style("border", "1px solid #333")
    .style("border-radius", "5px")
    .style("pointer-events", "none")
    .style("opacity", 0);
  
  // Mouse event functions for tooltip
  function mouseover(d) {
    tooltip.style("opacity", 1);
    d3.select(this).style("stroke", "black");
  }
  function mousemove(d) {
    tooltip.html("Count: " + d.count)
      .style("left", (d3.event.pageX + 10) + "px")
      .style("top", (d3.event.pageY - 15) + "px");
  }
  function mouseleave(d) {
    tooltip.style("opacity", 0);
    d3.select(this).style("stroke", "none");
  }
  
  // Draw the heatmap cells (rectangles)
  svg.selectAll(".cell")
    .data(counts)
    .enter()
    .append("rect")
      .attr("class", "cell")
      .attr("x", function(d) { return x(d.status); })
      .attr("y", function(d) { return y(d.feature); })
      .attr("width", x.bandwidth())
      .attr("height", y.bandwidth())
      .style("fill", function(d) { 
        return colorScale(d.count); 
      })
      .style("opacity", 0.8)
      .on("mouseover", mouseover)
      .on("mousemove", mousemove)
      .on("mouseleave", mouseleave);
  
  // ---- Optional: Add a horizontal legend for the color scale ----
  var legendWidth = 150,
      legendHeight = 10;
  
  var legend = svg.append("g")
      .attr("class", "legend")
      .attr("transform", "translate(" + (width - legendWidth) + ", -40)");
  
  // Create a gradient for the legend
  var defs = svg.append("defs");
  var linearGradient = defs.append("linearGradient")
      .attr("id", "legend-gradient");
  
  linearGradient.attr("x1", "0%")
      .attr("y1", "0%")
      .attr("x2", "100%")
      .attr("y2", "0%");
  
//   Set the gradient stops to match your color scale
//   linearGradient.append("stop")
//       .attr("offset", "0%")
//       .attr("stop-color", "#ffffe0");
//   linearGradient.append("stop")
//       .attr("offset", "100%")
//       .attr("stop-color", "#b22222");
  linearGradient.append("stop")
      .attr("offset", "0%")
      .attr("stop-color", "#eff3ff");  // light blue
  linearGradient.append("stop")
      .attr("offset", "100%")
      .attr("stop-color", "#08519c");  // dark blue
  
  
  // Draw the legend rectangle
  legend.append("rect")
      .attr("width", legendWidth)
      .attr("height", legendHeight)
      .style("fill", "url(#legend-gradient)");
  
  // Create an axis for the legend scale
  var legendScale = d3.scaleLinear()
      .domain([0, maxCount])
      .range([0, legendWidth]);
  
  legend.append("g")
      .attr("transform", "translate(0," + legendHeight + ")")
      .call(d3.axisBottom(legendScale).ticks(5));
});
