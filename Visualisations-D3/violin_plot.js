// set the dimensions and margins of the graph
var margin = {top: 60, right: 60, bottom: 60, left: 150},
    width = 800 - margin.left - margin.right,
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
    console.log("All data rows:", data);
      // After reading data...
    var groupCount = d3.nest()
    .key(d => d.Clip_ID)
    .rollup(v => v.length)
    .entries(data);

    console.log("Rows per clip:", groupCount);

    

      // Convert timestamps to numeric
    data.forEach(function(d) {
        console.log("Raw Selected_Timestamp:", d.Selected_Timestamp);
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

  // Add the shape to this svg!
  svg
    .selectAll("myViolin")
    .data(sumstat)
    .enter()        // So now we are working group per group
    .append("g")
      .attr("transform", function(d){ return("translate(" + x(d.key) +" ,0)") } ) // Translation on the right to be at the group position
    .append("path")
        .datum(function(d){ return(d.value)})     // So now we are working density per density
        .style("stroke", "none")
        .style("fill","#69b3a2")
        .attr("d", d3.area()
            .x0(function(d){ return(xNum(-d[1])) } )
            .x1(function(d){ return(xNum(d[1])) } )
            .y(function(d){ return(y(d[0])) } )
            .curve(d3.curveCatmullRom)    // This makes the line smoother to give the violin appearance. Try d3.curveStep to see the difference
        )

})

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
