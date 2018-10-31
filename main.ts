const imdb = require('imdb-api')
const googleTrends = require('google-trends-api')
const helper = require('./helper')

let MovieData = [];

let handler = function(err){
	console.log(err);
}


//Get the movie data and parse each entry as a element in the array
MovieData = helper.read('MovieData').split('\n')
console.log(MovieData.length)//Should come out as 1568 and it does