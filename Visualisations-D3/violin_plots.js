// Set the dimensions and margins of the graph
var margin = {top: 100, right: 50, bottom: 60, left: 150},
    width = 900 - margin.left - margin.right,
    height = 700 - margin.top - margin.bottom;

// Append the svg object to the body of the page
var svg = d3.select("#my_dataviz")
  .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");

// Create a dictionary to store the real interval per clip
var clipIntervals = {};

// Read the data
d3.csv("real_experiment_results.csv", function(error, data) {
  if (error) {
      console.error("Error loading data: ", error);
      return;
  }
  console.log("All data rows (unfiltered):", data);

  // 1) Parse numeric values and store Model_Salient_Interval
  data.forEach(function(d) {
    d.Selected_Timestamp = +d.Selected_Timestamp;
    d.Confidence_Level = +d.Confidence_Level;

    if (d.Model_Salient_Interval) {
      var parts = d.Model_Salient_Interval.replace(" sec", "").split("-");
      var lower = +parts[0];
      var upper = +parts[1];
      clipIntervals[d.Clip_ID] = [lower, upper];
    }
  });

  // 2) Filter to keep ONLY the participant-TP rows
//   //    (Clip actually depressed + Confidence > 5 => participant calls it depressed)
//   data = data.filter(function(d) {
//     // "TP" or "FN" means the clip is actually depressed
//     var isActuallyDepressed = (d.Clip_Type === "TP" || d.Clip_Type === "FN");
//     var participantSaysDepressed = (d.Confidence_Level > 5);
//     return (isActuallyDepressed && participantSaysDepressed);
//   });

  // TN means: Clip is actually NOT depressed AND participant says NOT depressed
// "Clip is not depressed" => Clip_Type in ["TN", "FP"]
// "Participant says not depressed" => Confidence_Level <= 5
// data = data.filter(function(d) {
//     var isActuallyNotDepressed = (d.Clip_Type === "TN" || d.Clip_Type === "FP");
//     var participantSaysNotDepressed = (d.Confidence_Level <= 5);
//     return (isActuallyNotDepressed && participantSaysNotDepressed);
//   });

// FP means: Clip is actually NOT depressed BUT participant says depressed
// "Clip is not depressed" => Clip_Type in ["TN", "FP"]
// "Participant says depressed" => Confidence_Level > 5
data = data.filter(function(d) {
    var isActuallyNotDepressed = (d.Clip_Type === "TN" || d.Clip_Type === "FP");
    var participantSaysDepressed = (d.Confidence_Level > 5);
    return (isActuallyNotDepressed && participantSaysDepressed);
  });

// FN means: Clip is actually depressed BUT participant says NOT depressed
// "Clip is depressed" => Clip_Type in ["TP", "FN"]
// "Participant says not depressed" => Confidence_Level <= 5
// data = data.filter(function(d) {
//     var isActuallyDepressed = (d.Clip_Type === "TP" || d.Clip_Type === "FN");
//     var participantSaysNotDepressed = (d.Confidence_Level <= 5);
//     return (isActuallyDepressed && participantSaysNotDepressed);
//   });


  console.log("Data rows AFTER filtering for participant-TP:", data);

  // If there's no data after filtering, you'll get an empty plot
  // so you might want to handle that case:
  if (data.length === 0) {
    svg.append("text")
      .attr("x", width/2)
      .attr("y", height/2)
      .attr("text-anchor", "middle")
      .text("No clips match participant-TP criteria.");
    return;
  }

  // 3) Group count (optional)
  var groupCount = d3.nest()
    .key(d => d.Clip_ID)
    .rollup(v => v.length)
    .entries(data);
  console.log("Rows per clip (TP only):", groupCount);

  // 4) Extract the unique clip IDs (TP only)
  var allClips = Array.from(new Set(data.map(d => d.Clip_ID)));

  // 5) Y scale: 0 to 8.5 (adjust if your timestamps differ)
  var y = d3.scaleLinear()
    .domain([0, 8.5])
    .range([height, 0]);
  svg.append("g").call(d3.axisLeft(y));

  // 6) X scale: each clip is a band
  var x = d3.scaleBand()
    .range([0, width])
    .domain(allClips)
    .padding(0.05);
  svg.append("g")
    .attr("transform", "translate(0," + height + ")")
    .call(d3.axisBottom(x));

  // 7) Define kernel density estimator
  var kde = kernelDensityEstimator(kernelEpanechnikov(0.2), y.ticks(50));

  // 8) Compute the density for each clip
  var sumstat = d3.nest()
    .key(function(d) { return d.Clip_ID; })
    .rollup(function(d) {
      var input = d.map(function(g) { return g.Selected_Timestamp; });
      var density = kde(input);
      return density;
    })
    .entries(data);

  // 9) Find the maximum density value (for violin width scaling)
  var maxNum = 0;
  sumstat.forEach(function(s) {
    var kdeValues = s.value.map(function(a) { return a[1]; });
    var biggest = d3.max(kdeValues);
    if (biggest > maxNum) {
      maxNum = biggest;
    }
  });

  // 10) X scale for the width of each violin
  var xNum = d3.scaleLinear()
    .range([0, x.bandwidth()])
    .domain([-maxNum, maxNum]);

  // 11) Create one "violin group" per clip
  var violin = svg.selectAll(".violin")
    .data(sumstat)
    .enter()
    .append("g")
      .attr("class", "violin")
      .attr("transform", function(d) {
        return "translate(" + x(d.key) + ",0)";
      });

  // 12) Append the violin path
  violin.append("path")
    .datum(function(d) { return d.value; })  // d.value is the density array
    .style("stroke", "none")
    // .style("fill", "#69b3a2")
    .style("fill", "#aec2de")

    .attr("d", d3.area()
      .x0(function(d) { return xNum(-d[1]); })
      .x1(function(d) { return xNum(d[1]); })
      .y(function(d)  { return y(d[0]); })
      .curve(d3.curveCatmullRom)
    );

//   // 13) Append the red circle for the model’s salient midpoint
//   violin.append("circle")
//     .attr("cx", xNum(0))
//     .attr("cy", function(d) {
//       // "d" here is the parent datum => {key: Clip_ID, value: [...density...] }
//       var clipId = d.key;
//       var interval = clipIntervals[clipId];
//       if (interval) {
//         var mid = (interval[0] + interval[1]) / 2;
//         return y(mid);
//       }
//       return y(0); // fallback if no interval
//     })
//     .attr("r", 5)
//     .style("fill", "red");
    // Append the shape (circle for TP, triangle for FN) indicating the model’s salient midpoint
violin.each(function(d) {
    var clipId = d.key;
    var interval = clipIntervals[clipId];

    // Determine segment classification (TP or FN)
    var clipData = data.find(cd => cd.Clip_ID === clipId);
    var isTP = (clipData.Clip_Type === "TN");
    var isFN = (clipData.Clip_Type === "FP");

    if (interval) {
        var mid = (interval[0] + interval[1]) / 2;

        if (isTP) {
            // Yellow circle for TP segments
            d3.select(this).append("circle")
                .attr("cx", xNum(0))
                .attr("cy", y(mid))
                .attr("r", 6)
                .style("fill", "green")
                .style("stroke", "black");
        } else if (isFN) {
            // Red triangle for FN segments
            d3.select(this).append("path")
                .attr("transform", "translate(" + xNum(0) + "," + y(mid) + ")")
                .attr("d", d3.symbol().type(d3.symbolTriangle).size(100))
                .style("fill", "red")
                .style("stroke", "black");
        }
    }
});


  // 14) (Optional) Add a legend
  var legend = svg.append("g")
    .attr("transform", "translate(" + (width - 130) + "," + -60 + ")");

  // Distribution legend
  legend.append("rect")
    .attr("x", 0)
    .attr("y", 0)
    .attr("width", 15)
    .attr("height", 15)
    .style("fill", "#aec2de");

  legend.append("text")
    .attr("x", 25)
    .attr("y", 12)
    .style("font-size", "14px")
    .text("Distribution of responses");



// Legend: TP midpoint (yellow circle)
legend.append("circle")
  .attr("cx", 7)
  .attr("cy", 35)
  .attr("r", 6)
  .style("fill", "green")
  .style("stroke", "black");

legend.append("text")
  .attr("x", 25)
  .attr("y", 39)
  .style("font-size", "14px")
  .text("Model salient point (TN)");

// Legend: FN midpoint (red triangle)
legend.append("path")
  .attr("transform", "translate(7, 60)")
  .attr("d", d3.symbol().type(d3.symbolTriangle).size(100))
  .style("fill", "red")
  .style("stroke", "black");

legend.append("text")
  .attr("x", 25)
  .attr("y", 64)
  .style("font-size", "14px")
  .text("Model salient point (FP)");
//   // Midpoint legend
//   legend.append("circle")
//     .attr("cx", 7)
//     .attr("cy", 35)
//     .attr("r", 5)
//     .style("fill", "red");

//   legend.append("text")
//     .attr("x", 25)
//     .attr("y", 39)
//     .style("font-size", "14px")
//     .text("True midpoint");
});

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
