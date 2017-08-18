
'use strict';

window.chartColors = {
	red: 'rgb(255, 99, 132)',
	orange: 'rgb(255, 159, 64)',
	yellow: 'rgb(255, 205, 86)',
	green: 'rgb(75, 192, 192)',
	blue: 'rgb(54, 162, 235)',
	purple: 'rgb(153, 102, 255)',
	grey: 'rgb(201, 203, 207)'
};

class Controller {
	constructor() {
		this.input = document.getElementById("input")
		this.canvas = document.getElementById("main");
		this.context = this.canvas.getContext("2d");
		this.canvas.addEventListener("mousedown", this.onMouseDown.bind(this));
		this.canvas.addEventListener("mouseup", this.onMouseUp.bind(this));
		this.canvas.addEventListener("mousemove", this.onMouseMove.bind(this));
		this.drawMatrix();


	}

	onMouseDown(e) {
		this.canvas.style.cursor = 'default';
		this.drawing = true;
		this.prev = this.getPosition(e.clientX, e.clientY);
	}

	onMouseUp(e) {
		this.drawing = false;
		this.drawInput();
	}

	onMouseMove(e) {
		if(this.drawing == true) {
			var pos = this.getPosition(e.clientX, e.clientY);
			this.context.lineWidth = 16;
			this.context.lineCap = 'round';
			this.context.beginPath();
			this.context.moveTo(this.prev.x, this.prev.y);
			this.context.lineTo(pos.x, pos.y);
			this.context.closePath();
			this.context.stroke();
			this.prev = pos;
		}
	}

	getPosition(clientX, clientY) {
		var rect = this.canvas.getBoundingClientRect();
		return {
			x: clientX - rect.left,
			y: clientY - rect.top
		};
	}

	drawInput() {
		var ctx = this.input.getContext('2d');
		var img = new Image();
		img.onload = () => {
			var inputs = [];
			var small = document.createElement('canvas').getContext('2d');
			small.drawImage(img, 0, 0, img.width, img.height, 0, 0, 28, 28);
			var data = small.getImageData(0, 0, 28, 28).data;
			for (var i = 0; i < 28; i++) {
				for (var j = 0; j < 28; j++) {
					var n = 4 * (i * 28 + j);
					inputs[i * 28 + j] = (data[n + 0] + data[n + 1] + data[n + 2]) / 3;
					ctx.fillStyle = 'rgb(' + [data[n + 0], data[n + 1], data[n + 2]].join(',') + ')';
					ctx.fillRect(j * 5, i * 5, 5, 5);
				}
			}
			if (Math.min(...inputs) === 255) {
				return;
			}
			$.ajax({
				url: 'http://127.0.0.1:8080/',
				method: 'POST',
				contentType: 'application/x-www-form-urlencoded',
				data : {
					'data': JSON.stringify(inputs),
				},
				success: (data) => {
					var Xs = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
					var color = Chart.helpers.color
					var res = data.split(",")
					var softmax = []
					var maxIndex = -1
					var maxValue = -100
					res.forEach(function(item, index){
						if(maxValue < item) {
							maxValue = item
							maxIndex = index
						}
						softmax.push(parseFloat(item))
					})

					//window.myBar.update();
					var chart = document.getElementById("canvas").getContext("2d");
					if(window.myBar == null) {
						this.barChartData = {
							labels: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
							datasets: [{
								label: 'Softmax',
								backgroundColor: [],//color(window.chartColors.red).alpha(0.5).rgbString(),
								borderColor: window.chartColors.red,
								borderWidth: 1,
								data: softmax
							}]
						}
						for(var i = 0; i < 10; i++){
							if(i == maxIndex){
								this.barChartData.datasets[0].backgroundColor[i] = color(window.chartColors.green).alpha(0.5).rgbString();
							} else {
								this.barChartData.datasets[0].backgroundColor[i] = color(window.chartColors.red).alpha(0.5).rgbString();
							}
						}
						window.myBar = new Chart(chart, {
							type: 'bar',
							data: this.barChartData,
							options: {
								responsive: true,
								legend: {
									position: 'top',
								},
								title: {
									display: true,
									text: 'Output'
								}
							}
						})

					} else {
						this.barChartData.datasets.forEach(function(dataset) {
							dataset.data = softmax;
						});
						for(var i = 0; i < 10; i++){
							if(i == maxIndex){
								window.myBar.data.datasets[0].backgroundColor[i] = color(window.chartColors.green).alpha(0.5).rgbString();
							} else {
								window.myBar.data.datasets[0].backgroundColor[i] = color(window.chartColors.red).alpha(0.5).rgbString();
							}
						}
						window.myBar.update();
					}
					//document.getElementById("matrix").innerHTML = data;
				}
			});
		}
		img.src = this.canvas.toDataURL();
	}

	drawMatrix() {
		this.context.fillStyle = "#ffffff";
		this.context.fillRect(0,0,449,449);
		this.context.lineWidth = 1;
		this.context.strokeRect(0,0,449,449);
		this.context.lineWidth = 0.05;
		for(var i = 0; i < 27; ++i) {
			this.context.beginPath();
			this.context.moveTo(16 * (i + 1),0);
			this.context.lineTo(16 * (i + 1),16*28);
			this.context.closePath();
			this.context.stroke();

			this.context.beginPath();
			this.context.moveTo(0, 16 * (i + 1));
			this.context.lineTo(16 * 28, 16 * (i + 1));
			this.context.closePath();
			this.context.stroke();
		}
		this.drawInput();
		$('#output td').text('').removeClass('success');
	}

	clearCanvas() {
		this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
		this.drawMatrix();
	}
}


$(() => {

	var controller = new Controller();
	$('#clear').click(() => {
		controller.clearCanvas();
	});
});
