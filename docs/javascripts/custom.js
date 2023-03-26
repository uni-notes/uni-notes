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

window.addEventListener("load", function(){
  let next = document.querySelector(".md-footer__link--next").href;
  prefetch(next.toLowerCase(), "document");
});
