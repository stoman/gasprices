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
	  	return datum.name + " (" + datum.brand + "), " + datum.street + ", " + datum.post_code + " " + datum.place; 
	  },
	  source: engine.ttAdapter(),
	  templates: {
      	suggestion: Handlebars.compile('<p><strong>{{name}} ({{brand}})</strong>, {{street}}, {{post_code}} {{place}}</p>')
      }
	});
	jQuery('.typeahead').on('typeahead:selected', function (e, datum) {
      $.fn.fullpage.moveTo('predictions');
    });
});
