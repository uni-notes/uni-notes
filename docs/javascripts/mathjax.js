window.MathJax = {
	// i'm loading manually
	loader: {load: [
		'[tex]/color',
		'[tex]/cases',
		'[tex]/mhchem'
	]},
  tex: {
		packages: {
			'[+]': [
				'color',
				'cases',
				'mhchem'
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