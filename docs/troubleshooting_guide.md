# Troubleshooting Guide - Smart Traffic Management System

## Quick Fixes

### SUMO Issues

#### SUMO Not Found
```
Error: SUMO executable not found
```
**Solution:**
```bash
# Windows
set SUMO_HOME=C:\path\to\sumo
set PATH=%PATH%;%SUMO_HOME%\bin

# Linux/Mac
export SUMO_HOME=/path/to/sumo
export PATH=$PATH:$SUMO_HOME/bin
```

#### SUMO GUI Not Working
```
Error: sumo-gui not found
```
**Solution:**
```bash
# Check if sumo-gui exists
ls $SUMO_HOME/bin/sumo-gui

# If not, install GUI version
# Or use sumo with --gui option
sumo -c config_file --gui
```

### Python Issues

#### Module Not Found
```
Error: ModuleNotFoundError: No module named 'pandas'
```
**Solution:**
```bash
pip install -r requirements_ab_tests.txt
```

#### Import Errors
```
Error: ImportError: cannot import name 'ABTestRunner'
```
**Solution:**
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Add src to path
export PYTHONPATH=$PYTHONPATH:./src
```

### Configuration Issues

#### Config File Not Found
```
Error: Config file not found
```
**Solution:**
1. Check file path in configuration
2. Ensure file exists
3. Check file permissions

#### Network File Missing
```
Error: Network file not found
```
**Solution:**
1. Check network file path
2. Ensure network file exists
3. Verify network file format

## Detailed Troubleshooting

### Demo Launcher Issues

#### Demo Won't Start
1. Check SUMO installation
2. Verify configuration files
3. Check file permissions
4. Review error logs

#### Demo Runs But No GUI
1. Check if sumo-gui is available
2. Try running SUMO directly
3. Check display settings
4. Use headless mode as backup

#### Demo Crashes
1. Check SUMO version compatibility
2. Verify network file format
3. Check route file format
4. Review error messages

### A/B Testing Issues

#### Tests Won't Run
1. Check SUMO configuration
2. Verify Python packages
3. Check file permissions
4. Review error logs

#### No Results Generated
1. Check output directory permissions
2. Verify test configuration
3. Check SUMO execution
4. Review error messages

#### Statistical Analysis Fails
1. Check data format
2. Verify sample sizes
3. Check Python packages
4. Review error messages

## Common Error Messages

### SUMO Errors

#### "No option with the name 'duration' exists"
```
Error: On processing option '--duration': No option with the name 'duration' exists
```
**Solution:** Use `--end` instead of `--duration`
```bash
# Wrong
sumo -c config.sumocfg --duration 300

# Correct
sumo -c config.sumocfg --end 300
```

#### "Network file not found"
```
Error: Could not load network 'network.net.xml'
```
**Solution:** Check network file path and existence
```bash
# Check if file exists
ls -la sumo/networks/network.net.xml

# Check file path in config
grep -n "net-file" sumo/configs/demo/demo_baseline.sumocfg
```

#### "Route file not found"
```
Error: Could not load routes 'routes.rou.xml'
```
**Solution:** Check route file path and existence
```bash
# Check if file exists
ls -la sumo/routes/routes.rou.xml

# Check file path in config
grep -n "route-files" sumo/configs/demo/demo_baseline.sumocfg
```

### Python Errors

#### "Permission denied"
```
Error: PermissionError: [Errno 13] Permission denied
```
**Solution:** Check file permissions
```bash
# Check permissions
ls -la scripts/demo_launcher.py

# Fix permissions
chmod +x scripts/demo_launcher.py
```

#### "No such file or directory"
```
Error: FileNotFoundError: [Errno 2] No such file or directory
```
**Solution:** Check file paths and existence
```bash
# Check if file exists
ls -la path/to/file

# Check current directory
pwd
```

## Debug Mode

### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Verbose SUMO Output
```bash
sumo -c config.sumocfg --verbose
```

### Check SUMO Configuration
```bash
sumo -c config.sumocfg --help
```

## System Requirements

### Minimum Requirements
- **OS**: Windows 10, macOS 10.14, or Linux Ubuntu 18.04+
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **Python**: 3.8 or later
- **SUMO**: 1.24.0 or later

### Recommended Requirements
- **OS**: Windows 11, macOS 12+, or Linux Ubuntu 20.04+
- **RAM**: 16GB
- **Storage**: 10GB free space
- **Python**: 3.9 or later
- **SUMO**: 1.25.0 or later

## Performance Issues

### Slow Demo Performance
1. Reduce demo duration
2. Lower traffic density
3. Use headless mode
4. Close other applications

### Memory Issues
1. Increase system RAM
2. Close other applications
3. Use smaller network files
4. Reduce traffic density

### CPU Issues
1. Use headless mode
2. Reduce simulation step length
3. Close other applications
4. Use faster hardware

## Backup Plans

### If SUMO Fails
1. Use pre-recorded videos
2. Show screenshots
3. Present static results
4. Use alternative simulator

### If Python Fails
1. Use direct SUMO commands
2. Show configuration files
3. Present static results
4. Use alternative tools

### If Network Fails
1. Use local files only
2. Show offline demos
3. Present static results
4. Use backup materials

## Getting Help

### Check Logs
```bash
# Check demo launcher logs
cat logs/demo_launcher.log

# Check A/B test logs
cat logs/ab_tests.log

# Check system logs
cat logs/system.log
```

### Contact Support
1. Check this troubleshooting guide
2. Review error messages
3. Check system requirements
4. Contact development team

### Report Issues
1. Describe the problem
2. Include error messages
3. Provide system information
4. Include log files

## Prevention

### Regular Maintenance
1. Update SUMO regularly
2. Update Python packages
3. Check file permissions
4. Clean up log files

### Testing
1. Test demos before presentation
2. Have backup plans ready
3. Practice troubleshooting
4. Keep spare hardware

### Documentation
1. Keep this guide updated
2. Document new issues
3. Share solutions
4. Update procedures

---

**Remember**: Stay calm, check the basics first, and have backup plans ready!

**Good luck with your SIH presentation!** 🚀
