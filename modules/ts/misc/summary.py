#!/usr/bin/env python
""" Format performance test results and compare metrics between test runs

Performance data is stored in the GTest log file created by performance tests. Default name is
`test_details.xml`. It can be changed with the `--gtest_output=xml:<location>/<filename>.xml` test
option. See https://github.com/opencv/opencv/wiki/HowToUsePerfTests for more details.

This script allows to compare performance data collected during separate test runs and present it in
a text, Markdown or HTML table.

### Major options

-o FMT, --output=FMT        - output format ('txt', 'html', 'markdown', 'tabs' or 'auto')
-f REGEX, --filter=REGEX    - regex to filter tests
-m NAME, --metric=NAME      - output metric
-u UNITS, --units=UNITS     - units for output values (s, ms (default), us, ns or ticks)

### Example

./summary.py -f LUT.*640 core1.xml core2.xml

Geometric mean (ms)

            Name of Test              core1  core2   core2
                                                       vs
                                                     core1
                                                   (x-factor)
LUT::OCL_LUTFixture::(640x480, 8UC1)  2.278  0.737    3.09
LUT::OCL_LUTFixture::(640x480, 32FC1) 2.622  0.805    3.26
LUT::OCL_LUTFixture::(640x480, 8UC4)  19.243 3.624    5.31
LUT::OCL_LUTFixture::(640x480, 32FC4) 21.254 4.296    4.95
LUT::SizePrm::640x480                 2.268  0.687    3.30
"""

import testlog_parser, sys, os, xml, glob, re
from table_formatter import *
from optparse import OptionParser

numeric_re = re.compile(r"(\d+)")
cvtype_re = re.compile(r"(8U|8S|16U|16S|32S|32F|64F)C(\d{1,3})")
cvtypes = { '8U': 0, '8S': 1, '16U': 2, '16S': 3, '32S': 4, '32F': 5, '64F': 6 }

convert = lambda text: int(text) if text.isdigit() else text
keyselector = lambda a: cvtype_re.sub(lambda match: " " + str(cvtypes.get(match.group(1), 7) + (int(match.group(2))-1) * 8) + " ", a)
alphanum_keyselector = lambda key: [ convert(c) for c in numeric_re.split(keyselector(key)) ]

def getSetName(tset, idx, columns, short = True):
    if columns and len(columns) > idx:
        prefix = columns[idx]
    else:
        prefix = None
    if short and prefix:
        return prefix
    name = tset[0].replace(".xml","").replace("_", "\n")
    if prefix:
        return prefix + "\n" + ("-"*int(len(max(prefix.split("\n"), key=len))*1.5)) + "\n" + name
    return name

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:\n", os.path.basename(sys.argv[0]), "<log_name1>.xml [<log_name2>.xml ...]", file=sys.stderr)
        exit(0)

    parser = OptionParser()
    parser.add_option("-o", "--output", dest="format", help="output results in text format (can be 'txt', 'html', 'markdown', 'tabs' or 'auto' - default)", metavar="FMT", default="auto")
    parser.add_option("-m", "--metric", dest="metric", help="output metric", metavar="NAME", default="gmean")
    parser.add_option("-u", "--units", dest="units", help="units for output values (s, ms (default), us, ns or ticks)", metavar="UNITS", default="ms")
    parser.add_option("-f", "--filter", dest="filter", help="regex to filter tests", metavar="REGEX", default=None)
    parser.add_option("", "--module", dest="module", default=None, metavar="NAME", help="module prefix for test names")
    parser.add_option("", "--columns", dest="columns", default=None, metavar="NAMES", help="comma-separated list of column aliases")
    parser.add_option("", "--no-relatives", action="store_false", dest="calc_relatives", default=True, help="do not output relative values")
    parser.add_option("", "--with-cycles-reduction", action="store_true", dest="calc_cr", default=False, help="output cycle reduction percentages")
    parser.add_option("", "--with-score", action="store_true", dest="calc_score", default=False, help="output automatic classification of speedups")
    parser.add_option("", "--progress", action="store_true", dest="progress_mode", default=False, help="enable progress mode")
    parser.add_option("", "--regressions", dest="regressions", default=None, metavar="LIST", help="comma-separated custom regressions map: \"[r][c]#current-#reference\" (indexes of columns are 0-based, \"r\" - reverse flag, \"c\" - color flag for base data)")
    parser.add_option("", "--show-all", action="store_true", dest="showall", default=False, help="also include empty and \"notrun\" lines")
    parser.add_option("", "--match", dest="match", default=None)
    parser.add_option("", "--match-replace", dest="match_replace", default="")
    parser.add_option("", "--regressions-only", dest="regressionsOnly", default=None, metavar="X-FACTOR", help="show only tests with performance regressions not")
    parser.add_option("", "--intersect-logs", dest="intersect_logs", default=False, help="show only tests present in all log files")
    parser.add_option("", "--show_units", action="store_true", dest="show_units", help="append units into table cells")
    (options, args) = parser.parse_args()

    options.generateHtml = detectHtmlOutputType(options.format)
    if options.metric not in metrix_table:
        options.metric = "gmean"
    if options.metric.endswith("%") or options.metric.endswith("$"):
        options.calc_relatives = False
        options.calc_cr = False
    if options.columns:
        options.columns = [s.strip().replace("\\n", "\n") for s in options.columns.split(",")]

    if options.regressions:
        assert not options.progress_mode, 'unsupported mode'

        def parseRegressionColumn(s):
            """ Format: '[r][c]<uint>-<uint>' """
            reverse = s.startswith('r')
            if reverse:
                s = s[1:]
            addColor = s.startswith('c')
            if addColor:
                s = s[1:]
            parts = s.split('-', 1)
            link = (int(parts[0]), int(parts[1]), reverse, addColor)
            assert link[0] != link[1]
            return link

        options.regressions = [parseRegressionColumn(s) for s in options.regressions.split(',')]

    show_units = options.units if options.show_units else None

    # expand wildcards and filter duplicates
    files = []
    seen = set()
    for arg in args:
        if ("*" in arg) or ("?" in arg):
            flist = [os.path.abspath(f) for f in glob.glob(arg)]
            flist = sorted(flist, key= lambda text: str(text).replace("M", "_"))
            files.extend([ x for x in flist if x not in seen and not seen.add(x)])
        else:
            fname = os.path.abspath(arg)
            if fname not in seen and not seen.add(fname):
                files.append(fname)

    # read all passed files
    test_sets = []
    for arg in files:
        try:
            tests = testlog_parser.parseLogFile(arg)
            if options.filter:
                expr = re.compile(options.filter)
                tests = [t for t in tests if expr.search(str(t))]
            if options.match:
                tests = [t for t in tests if t.get("status") != "notrun"]
            if tests:
                test_sets.append((os.path.basename(arg), tests))
        except IOError as err:
            sys.stderr.write("IOError reading \"" + arg + "\" - " + str(err) + os.linesep)
        except xml.parsers.expat.ExpatError as err:
            sys.stderr.write("ExpatError reading \"" + arg + "\" - " + str(err) + os.linesep)

    if not test_sets:
        sys.stderr.write("Error: no test data found" + os.linesep)
        quit()

    setsCount = len(test_sets)

    if options.regressions is None:
        reference = -1 if options.progress_mode else 0
        options.regressions = [(i, reference, False, True) for i in range(1, len(test_sets))]

    for link in options.regressions:
        (i, ref, reverse, addColor) = link
        assert i >= 0 and i < setsCount
        assert ref < setsCount

    # find matches
    test_cases = {}

    name_extractor = lambda name: str(name)
    if options.match:
        reg = re.compile(options.match)
        name_extractor = lambda name: reg.sub(options.match_replace, str(name))

    for i in range(setsCount):
        for case in test_sets[i][1]:
            name = name_extractor(case)
            if options.module:
                name = options.module + "::" + name
            if name not in test_cases:
                test_cases[name] = [None] * setsCount
            test_cases[name][i] = case

    # build table
    getter = metrix_table[options.metric][1]
    getter_score = metrix_table["score"][1] if options.calc_score else None
    getter_p = metrix_table[options.metric + "%"][1] if options.calc_relatives else None
    getter_cr = metrix_table[options.metric + "$"][1] if options.calc_cr else None
    tbl = table('%s (%s)' % (metrix_table[options.metric][0], options.units), options.format)

    # header
    tbl.newColumn("name", "Name of Test", align = "left", cssclass = "col_name")
    for i in range(setsCount):
        tbl.newColumn(str(i), getSetName(test_sets[i], i, options.columns, False), align = "center")

    def addHeaderColumns(suffix, description, cssclass):
        for link in options.regressions:
            (i, ref, reverse, addColor) = link
            if reverse:
                i, ref = ref, i
            current_set = test_sets[i]
            current = getSetName(current_set, i, options.columns)
            if ref >= 0:
                reference_set = test_sets[ref]
                reference = getSetName(reference_set, ref, options.columns)
            else:
                reference = 'previous'
            tbl.newColumn(str(i) + '-' + str(ref) + suffix, '%s\nvs\n%s\n(%s)' % (current, reference, description), align='center', cssclass=cssclass)

    if options.calc_cr:
        addHeaderColumns(suffix='$', description='cycles reduction', cssclass='col_cr')
    if options.calc_relatives:
        addHeaderColumns(suffix='%', description='x-factor', cssclass='col_rel')
    if options.calc_score:
        addHeaderColumns(suffix='S', description='score', cssclass='col_name')

    # rows
    prevGroupName = None
    needNewRow = True
    lastRow = None
    for name in sorted(test_cases.keys(), key=alphanum_keyselector):
        cases = test_cases[name]
        if needNewRow:
            lastRow = tbl.newRow()
            if not options.showall:
                needNewRow = False
        tbl.newCell("name", name)

        groupName = next(c for c in cases if c).shortName()
        if groupName != prevGroupName:
            prop = lastRow.props.get("cssclass", "")
            if "firstingroup" not in prop:
                lastRow.props["cssclass"] = prop + " firstingroup"
            prevGroupName = groupName

        for i in range(setsCount):
            case = cases[i]
            if case is None:
                if options.intersect_logs:
                    needNewRow = False
                    break
                tbl.newCell(str(i), "-")
            else:
                status = case.get("status")
                if status != "run":
                    tbl.newCell(str(i), status, color="red")
                else:
                    val = getter(case, cases[0], options.units)
                    if val:
                        needNewRow = True
                    tbl.newCell(str(i), formatValue(val, options.metric, show_units), val)

        if needNewRow:
            for link in options.regressions:
                (i, reference, reverse, addColor) = link
                if reverse:
                    i, reference = reference, i
                tblCellID = str(i) + '-' + str(reference)
                case = cases[i]
                if case is None:
                    if options.calc_relatives:
                        tbl.newCell(tblCellID + "%", "-")
                    if options.calc_cr:
                        tbl.newCell(tblCellID + "$", "-")
                    if options.calc_score:
                        tbl.newCell(tblCellID + "$", "-")
                else:
                    status = case.get("status")
                    if status != "run":
                        tbl.newCell(str(i), status, color="red")
                        if status != "notrun":
                            needNewRow = True
                        if options.calc_relatives:
                            tbl.newCell(tblCellID + "%", "-", color="red")
                        if options.calc_cr:
                            tbl.newCell(tblCellID + "$", "-", color="red")
                        if options.calc_score:
                            tbl.newCell(tblCellID + "S", "-", color="red")
                    else:
                        val = getter(case, cases[0], options.units)
                        def getRegression(fn):
                            if fn and val:
                                for j in reversed(range(i)) if reference < 0 else [reference]:
                                    r = cases[j]
                                    if r is not None and r.get("status") == 'run':
                                        return fn(case, r, options.units)
                        valp = getRegression(getter_p) if options.calc_relatives or options.progress_mode else None
                        valcr = getRegression(getter_cr) if options.calc_cr else None
                        val_score = getRegression(getter_score) if options.calc_score else None
                        if not valp:
                            color = None
                        elif valp > 1.05:
                            color = 'green'
                        elif valp < 0.95:
                            color = 'red'
                        else:
                            color = None
                        if addColor:
                            if not reverse:
                                tbl.newCell(str(i), formatValue(val, options.metric, show_units), val, color=color)
                            else:
                                r = cases[reference]
                                if r is not None and r.get("status") == 'run':
                                    val = getter(r, cases[0], options.units)
                                    tbl.newCell(str(reference), formatValue(val, options.metric, show_units), val, color=color)
                        if options.calc_relatives:
                            tbl.newCell(tblCellID + "%", formatValue(valp, "%"), valp, color=color, bold=color)
                        if options.calc_cr:
                            tbl.newCell(tblCellID + "$", formatValue(valcr, "$"), valcr, color=color, bold=color)
                        if options.calc_score:
                            tbl.newCell(tblCellID + "S", formatValue(val_score, "S"), val_score, color = color, bold = color)

    if not needNewRow:
        tbl.trimLastRow()

    if options.regressionsOnly:
        for r in reversed(range(len(tbl.rows))):
            for i in range(1, len(options.regressions) + 1):
                val = tbl.rows[r].cells[len(tbl.rows[r].cells) - i].value
                if val is not None and val < float(options.regressionsOnly):
                    break
            else:
                tbl.rows.pop(r)

    # output table
    if options.generateHtml:
        if options.format == "moinwiki":
            tbl.htmlPrintTable(sys.stdout, True)
        else:
            htmlPrintHeader(sys.stdout, "Summary report for %s tests from %s test logs" % (len(test_cases), setsCount))
            tbl.htmlPrintTable(sys.stdout)
            htmlPrintFooter(sys.stdout)
    else:
        tbl.consolePrintTable(sys.stdout)

    if options.regressionsOnly:
        sys.exit(len(tbl.rows))
