function upload() {
    var file_input = document.getElementById("upload");
    var reader = new FileReader();
    reader.onload = function (e) {
        $('#image').attr('src', e.target.result);
    }
    reader.readAsDataURL(file_input.files[0]);
}

function obfuscate() {
    var im = document.getElementById('image');
    var canvas = document.createElement('canvas');
    canvas.width = im.width;
    canvas.height = im.height;
    canvas.getContext('2d').drawImage(im, 0, 0, im.width, im.height);
    var data = canvas.getContext('2d').getImageData(0, 0, im.width, im.height).data;
    var inputs = [canvas.width, canvas.height];
    for (var i = 0; i < data.length; i+=4) {
       inputs.push([data[i], data[i+1], data[i+2]]);
    }
    $.ajax({
        // predict class for image
        url: '/illnoise/api/v0.1/obfuscate',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify(inputs),
        success: function(data) {
            $('#image').attr('src', data.obf_src);
        }
    });
}


$(function() {
    $('#orig_im').on('load', function () {
        var im = document.getElementById('orig_im');
        var canvas = document.createElement('canvas');
        canvas.width = im.width;
        canvas.height = im.height;
        canvas.getContext('2d').drawImage(im, 0, 0, im.width, im.height);
        var data = canvas.getContext('2d').getImageData(0, 0, im.width, im.height).data;
        var inputs = [];
        for (var i = 0; i < data.length; i+=4) {
           inputs.push([data[i], data[i+1], data[i+2]]);
        }
        $.ajax({
            // predict class for image
            url: '/illnoise/api/v0.1/predict',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(inputs),
            success: function(data) {
                var max = 0;
                var max_idx = 0;
                for (let i = 0; i < 10; i++) {
                    var val = Math.round(data.preds[0][i] * 1000);
                    if (val > max) {
                        max = val;
                        max_idx = i;
                    }
                    var n_digits = String(val).length;
                    for (let j = 0; j < 3 - n_digits; j++) {
                        val = '0' + val;
                    }
                    var text = '0.' + val;
                    if (val > 999) {
                        text = '1.000';
                    }
                    $('#preds tr').eq(i + 1).find('td').eq(0).text(text);
                }
            }
        });
        for (var j = 0; j < 10; j++) {
            $('#preds tr').eq(j + 1).find('td').eq(1).text('');
        }
        $('#obf_im').attr('src', '');
        $('#noise_im').attr('src', '');
    });

    $('#obfuscate_r').on('click', function() {
        var im = document.getElementById('orig_im');
        var canvas = document.createElement('canvas');
        canvas.width = im.width;
        canvas.height = im.height;
        canvas.getContext('2d').drawImage(im, 0, 0, im.width, im.height);
        var data = canvas.getContext('2d').getImageData(0, 0, im.width, im.height).data;
        var inputs = [];
        for (var i = 0; i < data.length; i+=4) {
           inputs.push([data[i], data[i+1], data[i+2]]);
        }
        $.ajax({
            // predict class for image
            url: '/illnoise/api/v0.1/obfuscate',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(inputs),
            success: function(data) {

                $('#obf_im').attr('src', data.obf_src);
                $('#noise_im').attr('src', data.noise_src);

                var max = 0;
                var max_idx = 0;
                for (let i = 0; i < 10; i++) {
                    var val = Math.round(data.preds[0][i] * 1000);
                    if (val > max) {
                        max = val;
                        max_idx = i;
                    }
                    var n_digits = String(val).length;
                    for (let j = 0; j < 3 - n_digits; j++) {
                        val = '0' + val;
                    }
                    var text = '0.' + val;
                    if (val > 999) {
                        text = '1.000';
                    }
                    $('#preds tr').eq(i + 1).find('td').eq(1).text(text);
                }
            }
        });
    });
});
