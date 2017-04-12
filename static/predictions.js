$(document).ready(function() {
	//load fullpage objects
	$('.fullpage').fullpage({
		anchors: ["predictions"]
	});
	
	//load charts
	createPredictionChart();
});

function createPredictionChart() {
	var margin = {top: 20, right: 10, bottom: 20, left: 50},
    width = $(".container").outerWidth() - margin.left - margin.right,
    height = 200 - margin.top - margin.bottom;
	//set the ranges
	var x = d3.scaleTime().range([0, width]);
	var y = d3.scaleLinear().range([height, 0]);
	//define the line
	var valueline = d3.line()
	    .x(function(d) { return x(d.date); })
	    .y(function(d) { return y(d.diesel); });
	//select the svg object to the body of the page
	var svg = d3.select("svg")
	    .attr("width", width + margin.left + margin.right)
	    .attr("height", height + margin.top + margin.bottom)
	    .append("g")
	    	.attr("transform", "translate(" + margin.left + "," + margin.top + ")");
	
	//load json
	d3.json("/api/prices/" + window.location.pathname.split("/")[2], function(error, data) {
		//TODO replace error handling
		if (error) return console.warn(error);

		//format the data
		data.forEach(function(d) {
		    d.date = new Date(d.date);
		    d.diesel = +d.diesel;
		});
		//scale the range of the data
		x.domain(d3.extent(data, function(d) { return d.date; }));
		y.domain([
		    d3.min(data, function(d) { return d.diesel; }) - 10,
		    d3.max(data, function(d) { return d.diesel; }) + 10
		]);
		//add the valueline path
		svg.append("path")
			.data([data])
			.attr("class", "line")
			.attr("d", valueline);
		//add the x axis
		svg.append("g")
			.attr("transform", "translate(0," + height + ")")
		    .call(d3.axisBottom(x));
		//add the y axis
		svg.append("g")
			.call(d3.axisLeft(y));
	});
}
