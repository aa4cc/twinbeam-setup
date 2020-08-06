$(function() {
  $('#but_enable').on('click', function () {
    $.get($SCRIPT_ROOT + '/start');        
  });
  $('#but_disable').on('click', function () {
    $.get($SCRIPT_ROOT + '/stop');        
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
    });
    return false;
  });
});
