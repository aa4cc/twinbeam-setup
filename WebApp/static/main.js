update_image_flag = 0;

$(function() {
  $('#collapseThree').on('show.bs.collapse', function () {
    update_image_flag = 1;
  })

  $('#collapseThree').on('hide.bs.collapse', function () {
    update_image_flag = 0;
  })  

  $('#but_enable').on('click', function () {
    $.get($SCRIPT_ROOT + '/start');        
  });

  $('#but_disable').on('click', function () {
    $.get($SCRIPT_ROOT + '/stop');        
  });

  $('#but_gen_restart').on('click', function () {
    $.get($SCRIPT_ROOT + '/service/restart/gen');
        
    setTimeout(function (){

      location.reload();
    
    }, 3000); 
  });  

  $('#but_gen_stop').on('click', function () {
    $.get($SCRIPT_ROOT + '/service/stop/gen');

    setTimeout(function (){

      location.reload();
    
    }, 3000); 
  });      

  $('#but_imgproc_restart').on('click', function () {
    $.get($SCRIPT_ROOT + '/service/restart/imgproc');

    setTimeout(function (){

      location.reload();
    
    }, 3000); 
  });    

  $('#but_imgproc_stop').on('click', function () {
    $.get($SCRIPT_ROOT + '/service/stop/imgproc');

    setTimeout(function (){

      location.reload();
    
    }, 3000); 
  });    

  $('#send_config').on('click', function () {
    var display_onoff = document.querySelector( 
      'input[name="display_onoff"]:checked'); 
    var savevideo_onoff = document.querySelector( 
      'input[name="savevideo_onoff"]:checked'); 
    var checkbox_value = "";
    $(":checkbox").each(function () {
        var ischecked = $(this).is(":checked");
        if (ischecked) {
            checkbox_value += $(this).val() + "|";
        }
    });

    $.getJSON($SCRIPT_ROOT + '/handle_data', {
        display_onoff:    display_onoff.value,
        savevideo_onoff:    savevideo_onoff.value,
        display_checkboxes: checkbox_value,
        restart: $('#restart').val(),
        img_r_thrs: $('#img_r_thrs').val(),
        img_g_thrs: $('#img_g_thrs').val()
      },
      function success(data) {
        console.log(data);
        // $('#files').html('');
        // for (var i = 0; i < data.files.length ; i++){
        //   $('#files').append('<tr> <td>' + data.files[i] + '</td> </tr>');
        // }        
      }
    );
    return false;
  });

  $('#but_files_refresh').on('click', refresh_files);

});

function download_file(filename){
  location.href = $SCRIPT_ROOT + '/download_file/' + filename;
}

function delete_file(filename){
  $.get($SCRIPT_ROOT + '/delete_file/' + filename);

  setTimeout(function (){

    refresh_files();
  
  }, 1000);   
}

function refresh_files() {
  $.getJSON($SCRIPT_ROOT + '/video_files', 
    function success(data) {
      $('#files').html('<tr> <th> Name</th> <th> Last Modification Time</th> <th> Size [MB]</th> <th> Action </th> </tr>');
      for (var i = 0; i < data.files.length ; i++){
        $('#files').append('<tr> <td>' + data.files[i][0] + '</td> <td>' + data.files[i][2] + '</td> <td>' + data.files[i][1] + '</td> <td><button type="button" class="btn btn-primary btn-sm" onclick="download_file(\''+data.files[i][0]+'\')" aria-label="Left Align"> <span class=" glyphicon glyphicon-download-alt " aria-hidden="true"></span> </button> <button type="button" class="btn btn-danger btn-sm" onclick="delete_file(\''+data.files[i][0]+'\')" aria-label="Left Align"> <span class=" glyphicon glyphicon-remove " aria-hidden="true"></span> </button> </td> </tr>');          
      }        
    }
  );

  return false;
}

function updateImage() {
  if (update_image_flag) {
    var image = $("#captured_image");
    src = image.attr("src");
    src = src.split("?")[0] + "?" + new Date().getTime();
    image.attr("src", src);    
  }
}

window.onload = function() {refresh_files();};
setInterval(updateImage, 250);