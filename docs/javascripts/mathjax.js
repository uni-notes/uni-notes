window.MathJax = {
	// i'm loading manually
	loader: {load: [
		'[tex]/color',
		'[tex]/cases',
		'[tex]/mhchem',
		'[tex]/textmacros'
	]},
  tex: {
		packages: {
			'[+]': [
				'color',
				'cases',
				'mhchem',
				'textmacros'
		]
	},
    inlineMath: [
			["\\(", "\\)"],
			["$", "$"]
		],
    displayMath: [
			["\\[", "\\]"],
			["$$", "$$"]
		],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => { 
  MathJax.typesetPromise()
})