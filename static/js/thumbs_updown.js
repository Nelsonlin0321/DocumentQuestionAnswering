$(document).ready(

    function(){

        $("#thumbsup_icon").click(

            function(){
                event.preventDefault();

                $("#thumbsup_icon").removeClass("outline");
                $("#thumbsdown_icon").addClass("outline");

                var question = $("#latest_question").text();
                var document = $("#latest_document").text();
                var answer = $("#latest_answer").text();

                $.post({

                    "url":"/",
                    "data":{
                        "log_question":question,
                        "log_document":document,
                        "log_answer":answer,
                        "feedback":"thumbsup",
                    },
    
                });


            }            

        );

    }

);


$(document).ready(

    function(){

        $("#thumbsdown_icon").click(

            function(){
                event.preventDefault();

                $("#thumbsdown_icon").removeClass("outline");
                $("#thumbsup_icon").addClass("outline");

                var question = $("#latest_question").text();
                var document = $("#latest_document").text();
                var answer = $("#latest_answer").text();

                $.post({

                    "url":"/",
                    "data":{
                        "log_question":question,
                        "log_document":document,
                        "log_answer":answer,
                        "feedback":"thumbsdown",
                    },
    
                });


            }            

        );

    }

);