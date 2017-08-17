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
			/*
			$.ajax({
				url: 'http://127.0.0.1:8080/',
				method: 'POST',
				contentType: 'application/x-www-form-urlencoded',
				data: inputs.toString(),
				success: (data) => {
					for (let i = 0; i < 2; i++) {
						var max = 0;
						var max_index = 0;
						for (let j = 0; j < 10; j++) {
							var value = Math.round(data.results[i][j] * 1000);
							if (value > max) {
								max = value;
								max_index = j;
							}
							var digits = String(value).length;
							for (var k = 0; k < 3 - digits; k++) {
								value = '0' + value;
							}
							var text = '0.' + value;
							if (value > 999) {
								text = '1.000';
							}
							$('#output tr').eq(j + 1).find('td').eq(i).text(text);
						}
						for (let j = 0; j < 10; j++) {
							if (j === max_index) {
								$('#output tr').eq(j + 1).find('td').eq(i).addClass('success');
							} else {
								$('#output tr').eq(j + 1).find('td').eq(i).removeClass('success');
							}
						}
					}
				}
			});
			*/
			$.ajax({
				url: 'http://127.0.0.1:8080/',
				method: 'POST',
				contentType: 'application/x-www-form-urlencoded',
				data : {
					'data': JSON.stringify(inputs),
				},
				success: (data) => {
					alert(data)
					$('#output tr').eq(1).find('td').eq(0).text(data);

					document.getElementById("matrix").innerHTML = JSON.stringify(inputs);
				}
			});
			/*
			$.ajax({
				url: 'http://127.0.0.1:8080/',
				method: 'GET',
				success: (data) => {
					alert(data)
				}
			});*/
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
