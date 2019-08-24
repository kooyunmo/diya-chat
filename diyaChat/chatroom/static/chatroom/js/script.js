$(document).ready(function () {
    $('#slides').superslides({
        animation: 'fade',
        play: 3000,
        pagination: false,
        inherit_height_from: '#slides',
        inherit_width_from: '#slides'
    });

    var typed = new Typed(".typed", {
        strings: ["The chatbot you've never seen", "Real AI chatbot learned with SOTA LMs", "Korean language is supported!!!", "Enjoy the conversation with AI"],
        typeSpeed: 70,
        loop: true,
        startDelay: 1000,
        showCursor: false
    });
});
