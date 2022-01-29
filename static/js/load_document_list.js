function LoadFile(){
    var oFrame = document.getElementById("frmFile");
    oFrame.onload = function(){
        var strRawContents = oFrame.contentWindow.document.body.childNodes[0].innerHTML;
        var document_names = strRawContents.split("\n");
        return document_names;
    }

    return oFrame.onload();
}

$(window).load(function(){

    var content = []

    var document_list  = LoadFile();
    
    for (var i = 0; i<document_list.length;i++){
        var document_name = document_list[i];
        content.push({title:document_name})
    }
    
    $('.ui.search')
      .search({
        source: content
      })
    ;

})

