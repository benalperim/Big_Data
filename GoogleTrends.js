const googleTrends = require('google-trends-api')
const helper = require('./helper')

let readData = [];
let outputData = '';
let TestArray = []
readData = helper.read('GoogleTrendSearch').split('\n')
console.log(readData.length);//Should come out as  1567 and it does





const trendData = async function(readData , outputData){

    for(let  i =0; i < 2; i++){



        let keyArray = readData[i].split(',' , 5)
        let timeFrame  =readData[i].split(',' , 6)
        let startTime = new Date(timeFrame[5].split('-')[0])
        let endTime = new Date(timeFrame[5].split('-')[1])
        console.log(keyArray)
        console.log(startTime)
        googleTrends
            .interestOverTime({
                keyword: keyArray,
                startTime,
                endTime,
                geo: 'US',
                //granularTimeResolution: true
            })
            .then((res) => {
                console.log(res);
            })
            .catch((err) => {
                console.log(err);
            });
    }
}(readData)