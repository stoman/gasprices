function loadPredictions(stid) {
	$(".section.predictions .content").hide();
	$(".section.predictions .loading").show();
	$.ajax({
		url: "/api/predictions/" + stid
	}).done(function(response) {
		$(".section.predictions .content").html(response).show();
		$(".section.predictions .loading").hide();
	});
}

$(document).ready(function() {
	//load fullpage objects
	$('.fullpage').fullpage({
		anchors: ["search", "predictions"]
	});
	
	//create gas station typeahead
	var engine = new Bloodhound({
      datumTokenizer: function (datum) {
        return Bloodhound.tokenizers.whitespace(datum.name);
	  },
	  queryTokenizer: Bloodhound.tokenizers.whitespace,
      prefetch: "/api/stations"
	});
	engine.initialize();
	
	$('.typeahead').typeahead({
	  hint: true,
	  highlight: true,
	  minLength: 1
	}, {
	  name: "engine",
	  displayKey: function(datum) {
	  	return datum.brand + ": " + datum.name + ", " + datum.street + ", " + datum.post_code + " " + datum.place; 
	  },
	  source: engine.ttAdapter(),
	  templates: {
      	suggestion: Handlebars.compile('<p><strong>{{brand}}: {{name}}</strong>, {{street}}, {{post_code}} {{place}}</p>')
      }
	});
	$('.typeahead').on('typeahead:selected', function (event, station) {
      $.fn.fullpage.moveTo('predictions');
      loadPredictions(station.id);
    });
});
