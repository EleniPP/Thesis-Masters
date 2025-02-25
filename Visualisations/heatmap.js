// // set the dimensions and margins of the graph
// var margin = {top: 80, right: 25, bottom: 30, left: 40},
//   width = 450 - margin.left - margin.right,
//   height = 450 - margin.top - margin.bottom;

// Increase margins so axis labels aren't cut off
var margin = { top: 60, right: 60, bottom: 60, left: 150 },
    width  = 800 - margin.left - margin.right,
    height = 600 - margin.top - margin.bottom;

// append the svg object to the body of the page
var svg = d3.select("#my_dataviz")
.append("svg")
  .attr("width", width + margin.left + margin.right)
  .attr("height", height + margin.top + margin.bottom)
.append("g")
  .attr("transform",
        "translate(" + margin.left + "," + margin.top + ")");

//Read the data
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
  
    // ---- New Heatmap Code Using Aggregated Data in "counts" ----

    // Compute unique groups (Clip_Type) and unique variables (Detected_Emotion) from "counts"
    var myGroups = d3.set(counts.map(function(d) { return d.Clip_Type; })).values();
    var myVars   = d3.set(counts.map(function(d) { return d.Detected_Emotion; })).values();

    // Build X scale (for Clip_Type)
    var x = d3.scaleBand()
    .range([0, width])
    .domain(myGroups)
    .padding(0.05);

    svg.append("g")
    .style("font-size", 15)
    .attr("transform", "translate(0," + height + ")")
    .call(d3.axisBottom(x).tickSize(0))
    .select(".domain").remove();

    // Build Y scale (for Detected_Emotion)
    var y = d3.scaleBand()
    .range([height, 0])
    .domain(myVars)
    .padding(0.05);

    svg.append("g")
    .style("font-size", 15)
    .call(d3.axisLeft(y).tickSize(0))
    .select(".domain").remove();

    // Build color scale
    // Since d3.scaleSequential is not available in D3 v4, we create our own using d3.scale.linear() and d3.interpolateInferno.
    var maxCount = d3.max(counts, function(d) { return d.Count; });
    var colorScale = d3.scaleLinear()
    .domain([0, maxCount])     // from 0 up to the highest count
    .range(["#ffffe0", "#b22222"]); // from light to dark (pick colors you like)

    var myColor = function(count) {
    var t = count / maxCount; // normalize count to [0,1]
    return d3.interpolateInferno(t);
    };

        // Example of a simple horizontal legend
    var legendWidth = 200;
    var legendHeight = 10;

    // Create a group for the legend
    var legend = svg.append("g")
    .attr("class", "legend")
    .attr("transform", "translate(" + (width - legendWidth) + "," + (-margin.top/2) + ")");

    // A gradient for the legend bar
    var gradient = svg.append("defs")
    .append("linearGradient")
        .attr("id", "legend-gradient")
        .attr("x1", "0%").attr("y1", "0%")
        .attr("x2", "100%").attr("y2", "0%");
    
    gradient.append("stop")
    .attr("offset", "0%")
    .attr("stop-color", "#ffffe0");  // same as colorScale(0)
    gradient.append("stop")
    .attr("offset", "100%")
    .attr("stop-color", "#b22222");  // same as colorScale(maxCount)

    // Draw the legend bar
    legend.append("rect")
    .attr("width", legendWidth)
    .attr("height", legendHeight)
    .style("fill", "url(#legend-gradient)");

    // Add an axis for the legend
    var legendScale = d3.scaleLinear()
    .domain([0, maxCount])
    .range([0, legendWidth]);

    legend.append("g")
    .attr("transform", "translate(0," + legendHeight + ")")
    .call(d3.axisBottom(legendScale).ticks(5));

    // Create a tooltip for interactivity
    var tooltip = d3.select("#my_dataviz")
    .append("div")
    .style("opacity", 0)
    .attr("class", "tooltip")
    .style("background-color", "white")
    .style("border", "solid")
    .style("border-width", "2px")
    .style("border-radius", "5px")
    .style("padding", "5px");

    var mouseover = function(d) {
    tooltip.style("opacity", 1);
    d3.select(this)
        .style("stroke", "black")
        .style("opacity", 1);
    };
    var mousemove = function(d) {
    tooltip.html("The exact value of<br>this cell is: " + d.Count)
        .style("left", (d3.mouse(this)[0] + 70) + "px")
        .style("top", (d3.mouse(this)[1]) + "px");
    };
    var mouseleave = function(d) {
    tooltip.style("opacity", 0);
    d3.select(this)
        .style("stroke", "none")
        .style("opacity", 0.8);
    };

    // Add the squares for the heatmap
    svg.selectAll("rect")
    .data(counts)
    .enter()
    .append("rect")
        .attr("x", function(d) { return x(d.Clip_Type); })
        .attr("y", function(d) { return y(d.Detected_Emotion); })
        .attr("rx", 4)
        .attr("ry", 4)
        .attr("width", x.bandwidth())
        .attr("height", y.bandwidth())
        .style("fill", function(d) { return colorScale(d.Count); })
        .style("stroke-width", 4)
        .style("stroke", "none")
        .style("opacity", 0.8)
    .on("mouseover", mouseover)
    .on("mousemove", mousemove)
    .on("mouseleave", mouseleave);});
