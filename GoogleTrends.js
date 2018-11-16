const googleTrends = require('google-trends-api')
const helper = require('./helper')

let readData = [];
let outputData = '';
let TestArray = []
readData = helper.read('GoogleTrendSearch').split('\n')
console.log(readData.length);//Should come out as  1567 and it does





const trendData = async function(readData , outputData){

    for(let  i =0; i < readData.length; i++){



        let keyArray = readData[2].split(',' , 5)
        let timeFrame  =readData[2].split(',' , 6)
        let startTime = new Date(timeFrame[5].split('-')[0])
        let endTime = new Date(timeFrame[5].split('-')[1])
        //console.log(keyArray)
        //console.log(startTime)
        let payload = await googleTrends
            .interestOverTime({
                keyword: keyArray,
                startTime,
                endTime,
                geo: 'US',
                //granularTimeResolution: true
            }).then((payload) => {
                //console.log(res)
                console.log(payload.split("averages\":")[1])
                
            }).catch((err) => {
                console.log(err);
            });
    }
}(readData)