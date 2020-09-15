//debugger;
//requirejs.config({paths: {d3: "https://d3js.org/d3.v4.min.js"}});
console.log("inside js");

require.config({paths: {d3: "http://localhost:8888/notebooks/d3/d3.v4.min"}});
require(["d3"], function(d3) {

  var margin = {top: 1, right: 1, bottom: 1, left: 1},
      width = 300 - margin.left - margin.right,
      height = 145 - margin.top - margin.bottom;


  d3.json("top.json", function(data) {

    function round(value, decimals) {
      return Number(Math.round(value+'e'+decimals)+'e-'+decimals);
    }

    window.fetchByLabels = function(labels, layers, heads, type) {
      return new Promise((resolve, reject) => {
          var callbacks = {
              iopub: {
                  output: (data) => resolve(data.content.text.trim())
              }
          };
          Jupyter.notebook.kernel.execute(`print(${python})`, callbacks);
      });
    };

    var tmpTokenizer = {8960: 'servants', 2049: 'force', 9602: 'Lithuania', 1669: 'period', 1425: 'age', 1556: 'With', 1817: 'leave', 6391: 'pregnant', 11037: 'Insurance', 1316: 'held', 2601: 'Law', 170: 'a', 1578: 'career', 8108: 'servant', 1201: 'years', 2740: 'cases', 6714: 'dismissed', 1210: 'three', 1470: 'public', 1219: 'during', 1482: 'children', 2250: 'Republic', 5965: 'reaches', 1103: 'the', 1104: 'of', 1105: 'and', 1235: 'until', 2516: 'Service', 1111: 'for', 1113: 'on',
    5986: 'insurance', 1250: 'work', 100: '[UNK]', 6245: 'benefits', 2790: 'provides', 1126: 'an', 2027: 'child', 113: '(', 114: ')', 1137: 'or', 117: ',', 24950: 'paragraph', 119: '.', 9467: 'specified', 17663: 'sickness',
  100: '[UNK]', 0: '[PAD]', 117: ',', 3854: 'Nations', 24950: 'paragraph', 123: '2', 1244: 'United', 1413: 'line', 2373: 'read', 6606: 'Fund', 13563: 'Property', 6534: 'Organization', 1367: '12', 4288: 'Children', 124: '3', 1291: 'World', 8500: 'Page', 1370: 'For'}


    var uniqueIndices = data.indices
      .reduce((prev, curr) => prev.concat(curr), [])
      .reduce((prev, curr) => prev.concat(curr), [])
      .filter((item, i, arr) => arr.indexOf(item) === i);

    debugger;

    var myColor = d3.scaleOrdinal().domain(uniqueIndices).range(d3.schemeCategory20b);//(["gold", "blue", "green", "yellow", "black", "grey", "darkgreen", "pink", "brown", "slateblue", "grey1", "orange"])

    window.decodeWord = function(index) {
      return tmpTokenizer[index];
    };

    var modelTable = d3.select('#model');
    var topkList = d3.select('#topk');

    var update = function(nofLayers, nofHeads, nofWords = 10) {

      var svg = modelTable
        .selectAll("tr")
        .data([...Array(nofHeads).keys()])
        .enter()
        .append("tr")
        .attr("head", function(d) {
          return d;
        })
        .selectAll("th")
        .data([...Array(nofLayers).keys()])
        .enter()
        .append("th")
        .attr("layer", function(d) {
          return d;
        })
        .append("svg");

      var svgData =
        svg
        .selectAll("rect")
        .data(function(d) {
          var head = parseInt(this.parentNode.parentNode.getAttribute("head"));
          var layer = d;
          var top = data.indices[layer][head].map(function(d, i) {
            return {"id": d, "value": data.values[layer][head][i]};
          });
          top.push({"id": "root"});
          var topWords = data.indices[layer][head];
          var topValues = data.values[layer][head];
          topWords.push("root");
          topValues.push("root");
          var root = d3.stratify()
            .id(function(d) { return d.id; })   // Name of the entity (column name is name in csv)
            .parentId(function(d, i) {
              return d.id === "root" ? "" : "root";
            })   // Name of the parent (column name is parent in csv)
          (top);
          root.sum(function(d,i) {
            return +d.value;
          });
          d3.treemap()
            .size([width, height])
            .padding(4)
            (root);
          return root.leaves();
        })
        .enter();
        svgData.append("rect")
          .attr('x', function (d) { return d.x0; })
          .attr('y', function (d) { return d.y0; })
          .attr('width', function (d) { return d.x1 - d.x0; })
          .attr('height', function (d) { return d.y1 - d.y0; })
          .style("stroke", "black")
          //.style("fill", "#69b3a2")
          .style("fill", function(d) {return myColor(d.id); })
        //.select(function() {return this.parentNode;})
        svgData.append("text")
          .attr('x', function (d) { return d.x0+10; })
          .attr('y', function (d) { return d.y0+20; })
          .text(function(d) {return window.decodeWord(d.id);})
          .attr("font-size", "15px")
          .attr("fill", "white");
        svgData.append("text")
          .attr('x', function (d) { return d.x0+10; })
          .attr('y', function (d) { return d.y0+40; })
          .text(function(d) {
            var prob = d.value;
            var percentage = round(prob, 3);
            console.log(percentage);
            return `${percentage}`;
          })
          .attr("font-size", "15px")
          .attr("fill", "white");


      // Head labels
      modelTable
        .insert("tr", "tr")
        .selectAll("th")
        .data([...Array(nofLayers)])
        .enter()
        .append("th")
        .html(function(_, d) {
          return `l<sub>${d}</sub>`;
        });

     //Layer labels
     modelTable
      .selectAll("tr")
      .insert("th", "th")
      .html(function() {
        var head = this.parentNode.getAttribute("head");
        if (head) {
          return `h<sub>${head}</sub>`;
        }
      });

      topkList
        .selectAll('tr')
        .data(data['indices'][nofLayers-1][nofHeads-1])
        .enter()
        .append('tr')
        .append('th')
        .text(function(d) {
          return window.decodeWord(d);
        })
        .style("background-color", function(d) {
          return myColor(d);
        })
        .style("color", function(d) {
          return "white";
        })
        .select(function() {
          return this.parentNode;
        })
        .append('th')
        .text(function(d, i) {
          var prob =  data['values'][nofLayers-1][nofHeads-1][i]
          var percentage = round(prob, 3) * 100
          return `${percentage}`;
        });
    };
    /*

    JSON response type (v1):

    {
      topk: [
        {
          "word": <word"
          "prob": 0-1
        },
        ...
      ]


    }

    JSON response type (v2):

    {
      layers: [
        0: {
            "word": <word>,
            "prob": 0 ... 1
          },
          { ... }
        1: { ... }
        ...
        12:
          {
            "word": <word>,
            "prob:" 0 ... 1
          }
      ],

      agg: [
        sum: [
          {...}
        ],
        last_layer: [
          {...}
        ]
      ],


    }
    */



    update(6, 12);

    // Generate and display some random table data

  });

});
