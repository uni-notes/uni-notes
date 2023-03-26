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
	let next_file = document.querySelector("a[rel='next']").innerText.split("\n").pop().replaceAll(" ", "_").toLowerCase() + "/";
  prefetch(next_file, "document")
});
