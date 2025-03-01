// set the dimensions and margins of the graph
var margin = {top: 100, right: 50, bottom: 60, left: 150},
    width = 900 - margin.left - margin.right,
    height = 700 - margin.top - margin.bottom;

// append the svg object to the body of the page
var svg = d3.select("#my_dataviz")
  .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");

// Create a dictionary to store the real interval per clip
var clipIntervals = {};

//Read the data
d3.csv("experiment_results.csv", function(error, data) {
    if (error) {
        console.error("Error loading data: ", error);
        return;
      }
    console.log("All data rows:", data);

    // For each row, parse the numeric values of the salient interval
    data.forEach(function(d) {
      // Convert selected timestamp to number
      d.Selected_Timestamp = +d.Selected_Timestamp;

      // Parse something like "4.4-4.5 sec" -> [4.4, 4.5]
      // (You only need to store it once per clip, but doing it on every row
      //  also works if the value is always the same for that clip.)
      if (d.Model_Salient_Interval) {
        // Remove " sec" and split on '-'
        var parts = d.Model_Salient_Interval.replace(" sec", "").split("-");
        var lower = +parts[0];
        var upper = +parts[1];

        // Store it in a dictionary keyed by Clip_ID
        clipIntervals[d.Clip_ID] = [lower, upper];
      }
    });


      // After reading data...
    var groupCount = d3.nest()
    .key(d => d.Clip_ID)
    .rollup(v => v.length)
    .entries(data);

    console.log("Rows per clip:", groupCount);

    

      // Convert timestamps to numeric
    data.forEach(function(d) {
        // console.log("Raw Selected_Timestamp:", d.Selected_Timestamp);
        d.Selected_Timestamp = +d.Selected_Timestamp;
    });

    // Extract the unique clip IDs
    var allClips = Array.from(new Set(data.map(d => d.Clip_ID)));
    
      // Y scale: 0 to 8.5 (adjust if your range differs)
    var y = d3.scaleLinear()
    .domain([0, 8.5])
    .range([height, 0]);
    svg.append("g").call(d3.axisLeft(y));
    
    // X scale: each clip is a "band"
    var x = d3.scaleBand()
    .range([0, width])
    .domain(allClips)
    .padding(0.05);
    svg.append("g")
    .attr("transform", "translate(0," + height + ")")
    .call(d3.axisBottom(x));

  // Features of density estimate
  var kde = kernelDensityEstimator(kernelEpanechnikov(0.2), y.ticks(50))

  // Compute the binning for each group of the dataset
  var sumstat = d3.nest()  // nest function allows to group the calculation per level of a factor
    .key(function(d) { return d.Clip_ID;})
    .rollup(function(d) {   // For each key..
      input = d.map(function(g) { return g.Selected_Timestamp;})    // Keep the variable called Sepal_Length
      density = kde(input)   // And compute the binning on it.
      return(density)
    })
    .entries(data)

  // What is the biggest value that the density estimate reach?
    var maxNum = 0;
    for ( i in sumstat ){
        var allBins = sumstat[i].value;
        var kdeValues = allBins.map(function(a){return a[1]});
        var biggest = d3.max(kdeValues);
        if (biggest > maxNum) { maxNum = biggest; }
    }


  // The maximum width of a violin must be x.bandwidth = the width dedicated to a group
  var xNum = d3.scaleLinear()
    .range([0, x.bandwidth()])
    .domain([-maxNum,maxNum])

  // First, create a "violin" group for each clip:
  var violin = svg
  .selectAll("myViolin")
  .data(sumstat)
  .enter()
  .append("g")
    .attr("transform", function(d){ 
      return "translate(" + x(d.key) + ",0)"; 
    });

  // Then, within each group, append the path (bound to the density array):
  violin
  .append("path")
    .datum(function(d){ return d.value; }) // "d" is {key, value}, we pass d.value to the path
    .style("stroke", "none")
    .style("fill","#69b3a2")
    .attr("d", d3.area()
      .x0(function(d){ return xNum(-d[1]); })
      .x1(function(d){ return xNum(d[1]); })
      .y(function(d){ return y(d[0]); })
      .curve(d3.curveCatmullRom)
    );

  // Finally, append the circle to the same group (NOT to the path),
  // so it retains the group-level data with .key:
  violin
  .append("circle")
    .attr("cx", xNum(0))
    .attr("cy", function(d){
      // Here, "d" is still {key: "Clip_1", value: [...density...]}
      var interval = clipIntervals[d.key];
      if (interval) {
        var mid = (interval[0] + interval[1]) / 2;
        return y(mid);
      }
      // fallback
      return y(0);
    })
    .attr("r", 5)
    .style("fill", "red");
  })


    //////////////////////////////////////
  // ADD A LEGEND
  //////////////////////////////////////

  // Create a legend group near the top-right (adjust x/y to your preference)
  var legend = svg.append("g")
  .attr("transform", "translate(" + (width - 110) + "," + -60 + ")");

  // 1) Legend item for the violin (distribution)
  legend.append("rect")
  .attr("x", 0)
  .attr("y", 0)
  .attr("width", 15)
  .attr("height", 15)
  .style("fill", "#69b3a2");

  legend.append("text")
  .attr("x", 25)
  .attr("y", 12)
  .style("font-size", "14px")
  .text("Distribution of responses");

  // 2) Legend item for the red circle (true midpoint)
  legend.append("circle")
  .attr("cx", 7)
  .attr("cy", 35)
  .attr("r", 5)
  .style("fill", "red");

  legend.append("text")
  .attr("x", 25)
  .attr("y", 39)
  .style("font-size", "14px")
  .text("True midpoint");
  // Add the shape to this svg!
//   svg
//     .selectAll("myViolin")
//     .data(sumstat)
//     .enter()        // So now we are working group per group
//     .append("g")
//       .attr("transform", function(d){ return("translate(" + x(d.key) +" ,0)") } ) // Translation on the right to be at the group position
//     .append("path")
//         .datum(function(d){ return(d.value)})     // So now we are working density per density
//         .style("stroke", "none")
//         .style("fill","#69b3a2")
//         .attr("d", d3.area()
//             .x0(function(d){ return(xNum(-d[1])) } )
//             .x1(function(d){ return(xNum(d[1])) } )
//             .y(function(d){ return(y(d[0])) } )
//             .curve(d3.curveCatmullRom)    // This makes the line smoother to give the violin appearance. Try d3.curveStep to see the difference
//         )
//     .append("circle")
//       // Center horizontally in the violin by using xNum(0)
//       .attr("cx", xNum(0))
//       // Map the midpoint of the real interval to the y scale
//       .attr("cy", function(d){
//         var interval = clipIntervals[d.key]; // e.g. [4.4, 4.5]
//         if (interval) {
//           var mid = (interval[0] + interval[1]) / 2;
//           return y(mid);
//         }
//         // Fallback if interval is missing:
//         return y(0);
//       })
//       .attr("r", 5)
//       .style("fill", "red");

// })

// 2 functions needed for kernel density estimate
function kernelDensityEstimator(kernel, X) {
  return function(V) {
    return X.map(function(x) {
      return [x, d3.mean(V, function(v) { return kernel(x - v); })];
    });
  };
}
function kernelEpanechnikov(k) {
  return function(v) {
    return Math.abs(v /= k) <= 1 ? 0.75 * (1 - v * v) / k : 0;
  };
}
