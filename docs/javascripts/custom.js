window.addEventListener("load", function(){
  
  const non_footer_links = document.querySelectorAll("a:not(footer a)");
  non_footer_links.forEach(function(link){
      link.textContent = link.textContent.replace(/[0-9]+[_ ]/, "")
  });

  const footer_directions = document.querySelectorAll(".md-footer__direction");
  footer_directions.forEach(function(direction) {
      direction.remove()
  })

  const footer_links = document.querySelectorAll("footer .md-ellipsis");
  footer_links.forEach(function(link){
      link.textContent = link.textContent.replace(/[0-9]+[_ ]/, "")
  });
  
  function prefetch(link, type)
  {
    const prefetch = document.createElement("link");

    prefetch.setAttribute("rel", "prefetch");
    prefetch.setAttribute("href", link);
    prefetch.setAttribute("as", type);
    if(type == "font" && link.includes("http")) // online font
      prefetch.setAttribute("crossorgin", true);

    document.head.appendChild(prefetch);
  }

  let next = document.querySelector(".md-footer__link--next").href;
  prefetch(next.toLowerCase(), "document");
  
  quicklink.listen();
  
});
